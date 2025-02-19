import torch
import torch.nn.functional as F
from espnet.nets.pytorch_backend.nets_utils import mask_by_length, make_non_pad_mask


class CTCLoss(torch.nn.Module):
    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ctc = torch.nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=True
        )

    def forward(self, preds_logprob, tokens_gt, cqt_lens, token_lens_gt):
        preds_logprob = mask_by_length(preds_logprob, cqt_lens)
        preds_logprob = torch.swapaxes(
            preds_logprob, 0, 1
        ) 
        loss = self.ctc(preds_logprob, tokens_gt, cqt_lens, token_lens_gt)

        return loss


class CELoss(torch.nn.Module):
    def __init__(
        self,
        mask_padding: bool = True,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none", weight=class_weights)

    def forward(self, preds_logits, gt, lens):
        preds_dtype = preds_logits.dtype
        gt_dtype = gt.dtype
        max_len = max(preds_logits.size(1), gt.size(1))
        if preds_logits.size(1) < max_len:
            padding_size = max_len - preds_logits.size(1)
            preds_logits = F.pad(preds_logits, (0, 0, 0, padding_size))
            preds_logits = preds_logits.to(preds_dtype)
        elif gt.size(1) < max_len:
            padding_size = max_len - gt.size(1)
            gt = F.pad(gt, (0, padding_size))
            gt = gt.to(gt_dtype)

        lens.fill_(max_len)
        out_mask = make_non_pad_mask(lens, length_dim=1).to(preds_logits.device)
        loss = self.loss_func(preds_logits.swapaxes(1, 2), gt)
        loss = loss.masked_select(out_mask)
        return torch.mean(loss)


class CustomLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.2,
        mask_padding: bool = True,
        group_weights: torch.Tensor = torch.Tensor([1.0, 0.5, 1.0, 2.0, 2.0]),
    ):
        super().__init__()
        self.alpha = alpha
        self.ctc_loss_func = CTCLoss()
        self.ce_loss_func = CELoss()
        self.ce_loss_func_group = CELoss(class_weights=group_weights)

    def forward(
        self,
        group_logits,
        tech_logits,
        final_tech_logits,
        frame_lens_gt,
        padded_tokens_gt,
        token_lens_gt,
    ):

        group_dict = {
            0: [0],
            1: [1],
            2: [2, 3, 4],
            3: [5, 6, 7],
            4: [8, 9]
        }
        tech_dict = {
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
        inverted_dict = {value: key for key, values in tech_dict.items() for value in values}
        padded_tech_gt = [[inverted_dict[element] for element in row] for row in padded_tokens_gt.tolist()]
        padded_tech_gt = torch.tensor(padded_tech_gt)
        padded_tech_gt = padded_tech_gt.to(tech_logits.device)
        inverted_dict = {value: key for key, values in group_dict.items() for value in values}
        padded_group_gt = [[inverted_dict[element] for element in row] for row in padded_tech_gt.tolist()]
        padded_group_gt = torch.tensor(padded_group_gt)
        padded_group_gt = padded_group_gt.to(group_logits.device)

        group_loss = self.ce_loss_func_group(group_logits, padded_group_gt, token_lens_gt)
        tech_loss = self.ce_loss_func(tech_logits, padded_tech_gt, token_lens_gt)
        padded_tokens_gt = padded_tokens_gt.to(final_tech_logits.device)
        final_tech_loss = self.ce_loss_func(final_tech_logits, padded_tokens_gt, token_lens_gt)

        loss = 0.5*group_loss + 1*tech_loss + 1.5*final_tech_loss

        return loss
