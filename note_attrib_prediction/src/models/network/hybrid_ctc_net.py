import torch
import torch.nn as nn
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder as TD
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from src.models.decoder.decoder import (
    GreedyCTCDecoder,
    GreedyAutoregressiveDecoder,
    BeamSearchAutoregressiveDecoder,
)


class ConvStack(nn.Module):
    """Class of Convolution stack module.

    Args:
        input_size: Input dimension (number of frequency bins).,
        output_size: Output dimension.
        conv_kernel_size: Kernel size. Same kernel size is used in all convolution layers.
        conv1_out_ch: Output channels for the first convolution layer.
        conv1_stride: Stride for the first convolution layer.
        conv2_out_ch: Output channels for the second convolution layer.
        conv2_stride: Stride for the second convolution layer.
        conv3_out_ch: Output channels for the third convolution layer.
        conv3_stride: Stride for the third convolution layer.
        activation: Type of activation function after each layer.
        conv_dropout: Dropout rate after each convolution layer.
        fc_dropout: Dropout rate after the final FC layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        conv_kernel_size: int,
        conv1_out_ch: int,
        conv1_stride: int,
        conv2_out_ch: int,
        conv2_stride: int,
        conv3_out_ch: int,
        conv3_stride: int,
        activation: str,
        conv_dropout: float,
        fc_dropout: float,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # layer 1
            nn.Conv2d(
                1,
                conv1_out_ch,
                conv_kernel_size,
                stride=(1, conv1_stride),
                padding=conv_kernel_size // 2,
            ),
            nn.BatchNorm2d(conv1_out_ch),
            getattr(nn, activation)(),
            nn.Dropout(conv_dropout),
            # layer 2
            nn.Conv2d(
                conv1_out_ch,
                conv2_out_ch,
                conv_kernel_size,
                stride=(1, conv2_stride),
                padding=conv_kernel_size // 2,
            ),
            nn.BatchNorm2d(conv2_out_ch),
            getattr(nn, activation)(),
            # layer 3
            # nn.MaxPool2d((1, 2)),
            nn.Dropout(conv_dropout),
            nn.Conv2d(
                conv2_out_ch,
                conv3_out_ch,
                conv_kernel_size,
                stride=(1, conv3_stride),
                padding=conv_kernel_size // 2,
            ),
            nn.BatchNorm2d(conv3_out_ch),
            getattr(nn, activation)(),
            # nn.MaxPool2d((1, 2)),
            nn.Dropout(conv_dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(
                conv3_out_ch
                * (input_size // (conv1_stride * conv2_stride * conv3_stride)),
                output_size,
            ),
            getattr(nn, activation)(),
            nn.Dropout(fc_dropout),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        y = self.cnn(x)
        y = y.transpose(1, 2).flatten(-2)
        y = self.fc(y)
        return y


class Conformer(nn.Module):
    """Class of Conformer encoder.

    Args:
        conformer_params: Dictionary of Conformer hyperparameters.
    """

    def __init__(
        self,
        conformer_params: dict,
    ):
        super().__init__()
        self.hparams = conformer_params
        self.conformer = ConformerEncoder(**conformer_params)
        self.tempo_linear = nn.Linear(
            conformer_params["input_size"] + 1, conformer_params["input_size"]
        )

    def forward(self, x, ilens, tempos):
        tempos = torch.unsqueeze(torch.unsqueeze(tempos, 1), 1)
        tempos = tempos.expand(-1, x.shape[1], -1)
        x = torch.cat((x, tempos), -1)
        x = self.tempo_linear(x)
        memory, _, _ = self.conformer(x, ilens)
        return memory


class CTCOutputLayer(nn.Module):
    """Class of the CTC output layer.

    Args:
        input_size: The input size (output size of Conformer encoder).
        input_size: The output size (vocab size).
        dropout: Dropout rate before the linear layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float,
    ):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        y = self.output_layer(x)
        return y


class TransformerDecoder(nn.Module):
    """Class of the Transformer decoder module.

    Args:
        decoder_params: Dictionary of the Transformer decoder hyperparameters.
        vocab_size: The vocabulary size.
    """

    def __init__(
        self,
        decoder_params: dict,
        vocab_size: int,
    ):
        super().__init__()
        decoder_params["vocab_size"] = vocab_size
        self.transformer_decoder = TD(**decoder_params)
        self.n_layer = decoder_params["num_blocks"]
        self.n_head = decoder_params["attention_heads"]

    def forward(self, memory, mlens, tokens, tlens):
        # y: logits
        y, _ = self.transformer_decoder(memory, mlens, tokens, tlens)
        return y


class HybridCTCNet(nn.Module):
    """Class of the hybrid CTC-Attention model.

    Args:
        conv_stack: Convolution stack module.
        conformer: Conformer module.
        transformer_decoder: Partially initialized Transformer decoder module.
        ctc_output_layer: Partially initialized CTC output layer.
        vocab_size: Size of the vocabulary.
        beam_search: Whether to use beam search algorithm when inferencing.
        beam_size: The beam size. Only used when beam_search=True.
        max_inference_length: Maximum length until the Transformer decoder stops generating new tokens.
    """

    def __init__(
        self,
        conv_stack: ConvStack,
        conformer: Conformer,
        transformer_decoder: TD,
        ctc_output_layer: CTCOutputLayer,
        vocab,
        vocab_size: int,
        beam_search: bool,
        beam_size: int,
        max_inference_length: int,
    ):
        super().__init__()
        self.conv_stack = conv_stack
        self.conformer = conformer
        self.ctc_output_layer = ctc_output_layer(output_size=vocab_size)
        self.transformer_decoder = transformer_decoder(vocab_size=vocab_size)
        self.sos_idx = vocab.__getitem__("SOS_None")
        self.eos_idx = vocab.__getitem__("EOS_None")
        self.vocab_size = vocab_size
        self.max_inference_length = max_inference_length
        self.ctc_decoder = GreedyCTCDecoder()
        if beam_search:
            self.ar_decoder = BeamSearchAutoregressiveDecoder(
                max_inference_length, self.sos_idx, self.eos_idx, beam_size, vocab_size
            )
        else:
            self.ar_decoder = GreedyAutoregressiveDecoder(
                max_inference_length, self.sos_idx, self.eos_idx
            )

    def forward(
        self,
        padded_cqt,
        cqt_lens,
        tokens,
        token_lens,
        tempos,
    ):
        memory = self.conv_stack(padded_cqt)
        memory = self.conformer(memory, cqt_lens, tempos)
        ctc_preds_logprob = self.ctc_output_layer(memory)

        tokens_wsos = self.add_sos(tokens)
        mask = make_non_pad_mask(token_lens, length_dim=1).to(memory.device)
        tokens_wsos_woeos = (tokens_wsos[:, :-1] * mask).int()

        token_preds_logits = self.transformer_decoder(
            memory, cqt_lens, tokens_wsos_woeos, token_lens
        )

        return ctc_preds_logprob, token_preds_logits

    def inference(
        self,
        padded_cqt,
        cqt_lens,
        tempos,
    ):
        memory = self.conv_stack(padded_cqt)
        memory = self.conformer(memory, cqt_lens, tempos)
        ctc_preds_logprob = self.ctc_output_layer(memory)
        ctc_preds_tokens = self.ctc_decoder(ctc_preds_logprob, cqt_lens)

        preds_tokens = self.ar_decoder(memory, cqt_lens, self.transformer_decoder)

        return ctc_preds_tokens, preds_tokens

    def add_sos(self, tokens):
        # adding sos token to non-one-hot tokens
        sos = torch.zeros(len(tokens), 1, device=tokens.device)
        sos += self.sos_idx
        tokens_with_sos = torch.cat((sos, tokens), 1)
        return tokens_with_sos
