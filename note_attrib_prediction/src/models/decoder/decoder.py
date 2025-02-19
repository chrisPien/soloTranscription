import torch
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


class GreedyAutoregressiveDecoder:
    def __init__(self, max_inference_length, sos_idx, eos_idx):
        self.max_len = max_inference_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __call__(self, memory, ilens, transformer_decoder):
        """
        Given a hidden representation from the encoder, autoregressively generate the best probable tokens.

        Args:
            memory : The encoder output. Shape [batch_size, input_length, dim]
            ilens : Length of the input sequences. Shape [batch_size]
            transformer_decoder : Transformer decoder object.

        Returns:
            List[str]: The resulting transcript
        """
        batch_size = len(memory)
        device = memory.device
        decoder_in = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        decoder_in += self.sos_idx
        decoder_in_lens = torch.ones(batch_size, dtype=torch.long, device=device)
        continue_flag = torch.ones(batch_size, dtype=torch.bool, device=device)
        for _ in range(self.max_len):
            decoder_out = transformer_decoder(
                memory, ilens, decoder_in, decoder_in_lens
            )
            last_token = torch.argmax(decoder_out[:, -1].unsqueeze(1), dim=2)
            continue_flag = torch.logical_and(
                continue_flag, ~(last_token.squeeze(1) == self.eos_idx)
            )
            last_token[~continue_flag] = 0
            decoder_in = torch.cat((decoder_in, last_token), dim=1)
            decoder_in_lens[continue_flag] += 1

            if continue_flag.sum() == 0:
                break

        decoded_out = []
        for out in decoder_in:
            decoded_out.append(out[out != 0].tolist()[1:])

        return decoded_out


class BeamSearchAutoregressiveDecoder:
    def __init__(self, max_inference_length, sos_idx, eos_idx, beam_size, vocab_size):
        self.max_len = max_inference_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.alpha = 0.7

    def __call__(self, memory, ilens, decoder):
        """
        Performs beam search decoding with a transformer decoder.

        Args:
            memory : The encoder output.
            memory : The encoder output. Shape [batch_size, input_length, dim]
            ilens : Length of the input sequences. Shape [batch_size]

        Returns:
            List[str]: The resulting transcript
        """
        device = memory.device
        tgt = torch.tensor(
            [self.sos_idx], dtype=torch.long, device=device
        )  # starting token
        memory_mask = None

        # Initialize the beam
        beam = [(tgt, 0)]
        for i in range(self.max_len):
            candidates = []
            for seq, score in beam:
                if seq[-1] == self.eos_idx:
                    # End of sequence
                    candidates.append((seq, score))
                    continue

                logits = decoder(
                    memory,
                    ilens,
                    seq.unsqueeze(0),
                    torch.tensor([len(seq)], dtype=torch.long, device=device),
                )
                prob = F.softmax(logits[:, -1, :], dim=-1)
                log_prob = torch.log(prob)
                scores, indices = torch.topk(log_prob, self.beam_size, dim=-1)
                for j in range(self.beam_size):
                    candidate_seq = torch.cat([seq, indices[0, j].view(1)])
                    candidate_score = score + scores[0, j]
                    candidates.append((candidate_seq, candidate_score))

            # Sort candidates by score
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Prune the beam
            beam = []
            for j in range(min(len(candidates), self.beam_size)):
                seq, score = candidates[j]
                if seq[-1] == self.eos_idx:
                    # End of sequence
                    beam.append((seq, score))
                else:
                    beam.append((seq, score - self.alpha * (len(seq) - 1)))

        # Return the best sequence
        best_seq, best_score = beam[0]
        best_seq = [best_seq.tolist()[1:]]

        return best_seq
