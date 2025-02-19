import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.nets_utils import mask_by_length

class GreedyCTCDecoder:
    def __init__(self, blank=0):
        super().__init__()
        self.blank = blank

    def __call__(self, preds_logprob, lengths):
        """
        Given a sequence of log probability over labels, get the best path.

        Args:
            preds_logprob : Logit tensors. Shape [batch_size, num_seq, num_label].
            lengths : Length of the sequences. Shape [batch_size]

        Returns:
            List[str]: The resulting transcript
        """
        batch_tokens = torch.argmax(preds_logprob, dim=-1)  # [batch_size, num_seq]
        batch_tokens = mask_by_length(batch_tokens, lengths)
        decoded_out = []
        for tokens in batch_tokens:
            tokens = torch.unique_consecutive(tokens)
            tokens = [i for i in tokens.tolist() if i != self.blank]
            decoded_out.append(tokens)
        return decoded_out

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
                * ((input_size) // (conv1_stride * conv2_stride * conv3_stride)) + 3,
                output_size,
            ),
            getattr(nn, activation)(),
            nn.Dropout(fc_dropout),
        )
        

    def forward(self, x, frame_level_note_attribs, note_attribs):
        x = torch.unsqueeze(x, dim=1)
        y = self.cnn(x)
        y = y.transpose(1, 2).flatten(-2)     
        y = torch.cat((y, frame_level_note_attribs), -1)      
        y = self.fc(y)
        return y


class HierarchicalOutputLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        group_size: int,
        tech_size: int,
        final_tech_size: int,
        dropout: float,
    ):
        super().__init__()
        self.group_to_tech_map = {
            0: [0],
            1: [1],
            2: [2, 3, 4],
            3: [5, 6, 7],
            4: [8, 9]
        }
        self.tech_to_final_tech_map = {
            0: [0],
            1: [1, 19],
            2: [2, 3, 4],
            3: [5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 24],
            4: [13],
            5: [17],
            6: [15],
            7: [18],
            8: [16],
            9: [14, 23]
        }
        self.group_to_final_tech_map = {
            0: [0],
            1: [1, 19],
            2: [2, 3, 4],
            3: [5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 24],
            4: [13, 14, 23]
        }
        self.group_size = group_size + 1
        self.tech_size = tech_size + 1
        self.final_tech_size = final_tech_size + 1
        self.gruop_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, self.group_size),
        )
        self.tech_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, self.tech_size),
        )
        self.final_tech_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, self.final_tech_size),
        )

    def forward(self, x, note_size):
        group_logits = self.gruop_layer(x)
        group_pred = torch.argmax(group_logits, dim=-1)
        tech_logits = torch.zeros(x.size(0), x.size(1), self.tech_size, device=torch.device('cuda'))
        final_tech_logits = torch.zeros(x.size(0), x.size(1), self.final_tech_size, device=torch.device('cuda'))
        tech_layer_output = self.tech_layer(x)
        for group_idx in range(self.group_size):
            mask = (group_pred == group_idx).unsqueeze(-1)
            tech_indices = self.group_to_tech_map[group_idx]

            tech_logits[:, :, tech_indices] += mask.float() * tech_layer_output[:, :, tech_indices]

        # Predict final tech logits
        tech_pred = torch.argmax(tech_logits, dim=-1)
        final_tech_layer_output = self.final_tech_layer(x)

        # For each batch and sequence, map tech predictions to appropriate final tech logits
        for tech_idx in range(self.tech_size):
            mask = (tech_pred == tech_idx).unsqueeze(-1)
            final_tech_indices = self.tech_to_final_tech_map[tech_idx]

            final_tech_logits[:, :, final_tech_indices] += mask.float() * final_tech_layer_output[:, :, final_tech_indices]
            frame_level_final_tech_logits = final_tech_logits

        group_logits = nn.Upsample(size=note_size)(group_logits.permute(0, 2, 1)).permute(0, 2, 1)
        tech_logits = nn.Upsample(size=note_size)(tech_logits.permute(0, 2, 1)).permute(0, 2, 1)
        final_tech_logits = nn.Upsample(size=note_size)(final_tech_logits.permute(0, 2, 1)).permute(0, 2, 1)

        return group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits


class CNNNet(nn.Module):
    def __init__(
        self,
        conv_stack: ConvStack,
        hierarchical_output_layer: HierarchicalOutputLayer,
        max_inference_length: int,
    ):
        super().__init__()
        self.conv_stack = conv_stack
        self.hierarchical_output_layer = hierarchical_output_layer()
        self.max_inference_length = max_inference_length
        self.ctc_decoder = GreedyCTCDecoder()

    def forward(
        self,
        padded_cqt,
        frame_level_note_attribs,
        note_attribs,
    ):
        memory = self.conv_stack(padded_cqt, frame_level_note_attribs, note_attribs)
        group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits = self.hierarchical_output_layer(memory, note_attribs.shape[1])

        return group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits
    
    def inference(
        self,
        padded_cqt,
        cqt_lens,
        frame_level_note_attribs,
        note_attribs,
    ):
        memory = self.conv_stack(padded_cqt, frame_level_note_attribs, note_attribs)
        group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits = self.hierarchical_output_layer(memory, note_attribs.shape[1])
        group_preds = torch.argmax(group_logits, dim=-1)
        tech_preds = torch.argmax(tech_logits, dim=-1)
        final_tech_preds = torch.argmax(final_tech_logits, dim=-1)
        frame_level_final_tech_preds = torch.argmax(frame_level_final_tech_logits, dim=-1)

        return group_preds, tech_preds, final_tech_preds, frame_level_final_tech_preds
