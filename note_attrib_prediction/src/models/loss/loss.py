import torch
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
    ):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, preds_logits, gt, lens):
        out_mask = make_non_pad_mask(lens, length_dim=1).to(preds_logits.device)
        loss = self.loss_func(preds_logits.swapaxes(1, 2), gt)
        loss = loss.masked_select(out_mask)
        return torch.mean(loss)


class CustomLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.2,
        mask_padding: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.ctc_loss_func = CTCLoss()
        self.ce_loss_func = CELoss()

    def forward(
        self,
        ctc_preds_logprob,
        token_preds_logits,
        frame_lens_gt,
        padded_tokens_gt,
        token_lens_gt,
    ):
        token_mask = make_non_pad_mask(token_lens_gt, length_dim=1).to(
            token_preds_logits.device
        )
        ctc_loss = self.ctc_loss_func(
            ctc_preds_logprob, padded_tokens_gt, frame_lens_gt, token_lens_gt
        )
        ce_loss = self.ce_loss_func(token_preds_logits, padded_tokens_gt, token_lens_gt)

        loss = self.alpha * ctc_loss + ce_loss
        #loss = ce_loss

        return loss
