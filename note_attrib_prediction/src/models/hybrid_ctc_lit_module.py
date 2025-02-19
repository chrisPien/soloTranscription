from typing import Any, List, Optional
import torch
import os
import sys
sys.path.append(os.getcwd())
from matplotlib import pyplot as plt
import librosa
import pretty_midi
import numpy as np
import librosa.display
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import seaborn as sns
from pytorch_lightning import LightningModule
from torchmetrics.functional import word_error_rate
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from espnet.nets.pytorch_backend.nets_utils import mask_by_length


class HybridCTCLitModule(LightningModule):
    """LightningModule for automatic guitar transcription using hybrid CTC-Attention model.

    This module organizes the code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers and LR Schedulers (configure_optimizers)
        - Other miscellaneous functions (make pianoroll, plot, etc.)
    """

    def __init__(
        self,
        hybrid_ctc_net: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: any,
        tokenizer: any,
        vocab: any,
        vocab_size: int,
        output_dir: str,
        prediction_dir: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['loss_func'])

        # pass vocab size to initialize the partially initialized network
        self.network = hybrid_ctc_net(vocab=vocab, vocab_size=vocab_size)

        # loss function
        self.loss_func = loss_func

        # tokenizer
        self.tokenizer = tokenizer

        # dir for saving the results
        self.output_dir = output_dir

        # dir for saving prediction for tech_prediction
        self.prediction_dir = os.path.abspath(prediction_dir)

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # metrics container
        self.val_loss = MeanMetric()
        self.val_ctc_ter = MeanMetric()
        self.val_ter = MeanMetric()

        self.test_ctc_token_precision = MeanMetric()
        self.test_ctc_token_recall = MeanMetric()
        self.test_ctc_token_f1 = MeanMetric()
        self.test_tr_token_precision = MeanMetric()
        self.test_tr_token_recall = MeanMetric()
        self.test_tr_token_f1 = MeanMetric()
        self.test_ctc_ter = MeanMetric()
        self.test_tr_ter = MeanMetric()

    def forward(
        self,
        padded_cqt,
        cqt_lens,
        padded_tokens_gt,
        token_lens_gt,
        tempos,
    ):
        return self.network(
            padded_cqt, cqt_lens, padded_tokens_gt, token_lens_gt, tempos
        )

    def model_step(self, batch: dict):
        ctc_preds_logprob, tr_preds_logits = self.forward(
            batch["padded_cqt"],
            batch["cqt_lens"],
            batch["padded_tokens_gt"],
            batch["token_lens_gt"],
            batch["tempos"],
        )

        loss = self.loss_func(
            ctc_preds_logprob,
            tr_preds_logits,
            batch["cqt_lens"],
            batch["padded_tokens_gt"].long(),
            batch["token_lens_gt"],
        )

        return loss, ctc_preds_logprob, tr_preds_logits

    def model_inference(self, padded_cqt, cqt_lens, tempos):
        ctc_preds_tokens, tr_preds_tokens = self.network.inference(
            padded_cqt, cqt_lens, tempos
        )
        return ctc_preds_tokens, tr_preds_tokens

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    #     loss, ctc_preds_logprob, tr_preds_logits = self.model_step(batch)

    #     # update and log metrics
    #     self.val_loss(loss)
    #     self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    #     # ctc tokens
    #     preds_ctc_tokens = self.network.ctc_decoder(
    #         ctc_preds_logprob, batch["cqt_lens"]
    #     )

    #     # tokens
    #     tr_preds_tokens = torch.argmax(tr_preds_logits, dim=2)
    #     tr_preds_tokens = [
    #         tr_preds_tokens[i, : batch["token_lens_gt"][i]].tolist()
    #         for i in range(len(batch["token_lens_gt"]))
    #     ]
    #     gt_tokens = [
    #         batch["padded_tokens_gt"][i, : batch["token_lens_gt"][i]].tolist()
    #         for i in range(len(batch["token_lens_gt"]))
    #     ]

    #     self.val_ctc_ter(self.ter(preds_ctc_tokens, gt_tokens))
    #     self.val_ter(self.ter(tr_preds_tokens, gt_tokens))
    #     self.log(
    #         "val/ctc_ter", self.val_ctc_ter, on_step=False, on_epoch=True, prog_bar=True
    #     )
    #     self.log("val/ter", self.val_ter, on_step=False, on_epoch=True, prog_bar=True)

    #     return {"loss": loss}

    def on_test_epoch_start(self):
        self.tab_output_dir = os.path.join(self.output_dir, "tab")
        # self.figs_output_dir = os.path.join(self.output_dir, "figs")
        self.tokens_output_dir = os.path.join(self.output_dir, "tokens")
        if not os.path.exists(self.tab_output_dir):
            os.makedirs(self.tab_output_dir)
            # os.makedirs(self.figs_output_dir)
            os.makedirs(self.tokens_output_dir)

    def test_step(self, batch, batch_idx):
        ctc_preds_tokens, tr_preds_tokens = self.model_inference(
            batch["padded_cqt"],
            batch["cqt_lens"],
            batch["tempos"],
        )

        # tokens
        padded_tokens_gt = batch["padded_tokens_gt"]
        token_lens_gt = batch["token_lens_gt"]
        track_name_list = batch["track_name_list"]

        gt_tokens = [
            padded_tokens_gt[i, : token_lens_gt[i]].tolist()
            for i in range(len(padded_tokens_gt))
        ]

        # ctc_precision = torch.zeros(len(tr_preds_tokens))
        # ctc_recall = torch.zeros(len(tr_preds_tokens))
        # ctc_f1 = torch.zeros(len(tr_preds_tokens))

        # tr_precision = torch.zeros(len(tr_preds_tokens))
        # tr_recall = torch.zeros(len(tr_preds_tokens))
        # tr_f1 = torch.zeros(len(tr_preds_tokens))

        for i in range(len(tr_preds_tokens)):
            tempo = batch["tempos"][i].item()
            ctc_preds_tab = self.tokenizer.tokens_to_tab([ctc_preds_tokens[i]])
            ctc_preds_tab = [[int(element) for element in row] for row in ctc_preds_tab]
            tr_preds_tab = self.tokenizer.tokens_to_tab([tr_preds_tokens[i]])
            tr_preds_tab = [[int(element) for element in row] for row in tr_preds_tab]
            # ctc_preds_midi.tempo_changes[0].tempo = tempo
            # tr_preds_midi.tempo_changes[0].tempo = tempo
            ctc_preds_tab_path = os.path.join(
                self.tab_output_dir, track_name_list[i] + "_ctc_pred.tsv"
            )
            tr_preds_tab_path = os.path.join(
                self.tab_output_dir, track_name_list[i] + "_transformer_pred.tsv"
            )
            # convert tatum to pos dur

            # ctc_preds_tab_pos_dur = self.tatum_to_pos_dur(ctc_preds_tab)
            # with open(ctc_preds_tab_path, 'w') as f:
            #     for line in ctc_preds_tab_pos_dur:
            #         f.write(f"{line}\n")
            # f.close
            with open(ctc_preds_tab_path, 'w') as f:
                for line in ctc_preds_tab:
                    f.write(f"{line}\n")
            f.close
            # tr_preds_tab_pos_dur = self.tatum_to_pos_dur(tr_preds_tab)
            # with open(tr_preds_tab_path, 'w') as f:
            #     for line in tr_preds_tab_pos_dur:
            #         f.write(f"{line}\n")
            # f.close
            with open(tr_preds_tab_path, 'w') as f:
                for line in tr_preds_tab:
                    f.write(f"{line}\n")
            f.close
            gt_tab = self.tokenizer.tokens_to_tab([gt_tokens[i]])
            gt_tab = [[int(element) for element in row] for row in gt_tab]
            
            # gt_tab_pos_dur = self.tatum_to_pos_dur(gt_tab)
            gt_tab_path = os.path.join(
                self.tab_output_dir, track_name_list[i] + "_gt.tsv"
            )
            # with open(gt_tab_path, 'w') as f:
            #     for line in gt_tab_pos_dur:
            #         f.write(f"{line}\n")
            # f.close
            with open(gt_tab_path, 'w') as f:
                for line in gt_tab:
                    f.write(f"{line}\n")
            f.close

            ctc_preds_tokens_path = os.path.join(
                self.tokens_output_dir, track_name_list[i] + "_ctc_pred.tsv"
            )
            tr_preds_tokens_path = os.path.join(
                self.tokens_output_dir, track_name_list[i] + "_transformer_pred.tsv"
            )
            gt_tokens_path = os.path.join(
                self.tokens_output_dir, track_name_list[i] + "_gt.tsv"
            )

            with open(ctc_preds_tokens_path, 'w') as f:
                for line in ctc_preds_tokens[i]:
                    f.write(f"{line}\n")
            f.close
            with open(tr_preds_tokens_path, 'w') as f:
                for line in tr_preds_tokens[i]:
                    f.write(f"{line}\n")
            f.close
            with open(gt_tokens_path, 'w') as f:
                for line in gt_tokens[i]:
                    f.write(f"{line}\n")
            f.close

            prediction_path = os.path.join(
                self.prediction_dir, track_name_list[i] + ".npy"
            )
            np.save(prediction_path, np.array(tr_preds_tab))

        #     # convert the tokens into pianoroll
        #     ctc_preds_pianoroll = self.make_pianoroll(ctc_preds_tab_pos_dur, tempo)
        #     tr_preds_pianoroll = self.make_pianoroll(tr_preds_tab_pos_dur, tempo)
        #     gt_pianoroll = self.make_pianoroll(gt_tab_pos_dur, tempo)

        #     # make even length
        #     if len(ctc_preds_pianoroll) > len(gt_pianoroll):
        #         ctc_preds_pianoroll = ctc_preds_pianoroll[: len(gt_pianoroll)]
        #     elif len(ctc_preds_pianoroll) < len(gt_pianoroll):
        #         zeros = torch.zeros_like(gt_pianoroll)
        #         zeros[: len(ctc_preds_pianoroll)] = ctc_preds_pianoroll
        #         ctc_preds_pianoroll = zeros

        #     if len(tr_preds_pianoroll) > len(gt_pianoroll):
        #         tr_preds_pianoroll = tr_preds_pianoroll[: len(gt_pianoroll)]
        #     elif len(tr_preds_pianoroll) < len(gt_pianoroll):
        #         zeros = torch.zeros_like(gt_pianoroll)
        #         zeros[: len(tr_preds_pianoroll)] = tr_preds_pianoroll
        #         tr_preds_pianoroll = zeros

        #     # plot
        #     self.plot_output(
        #         gt_pianoroll,
        #         ctc_preds_pianoroll,
        #         tr_preds_pianoroll,
        #         track_name_list[i],
        #     )

        #     ctc_p, ctc_r, ctc_f = self.get_precision_recall_f1(
        #         ctc_preds_pianoroll, gt_pianoroll
        #     )
        #     tr_p, tr_r, tr_f = self.get_precision_recall_f1(
        #         tr_preds_pianoroll, gt_pianoroll
        #     )

        #     ctc_precision[i] = ctc_p
        #     ctc_recall[i] = ctc_r
        #     ctc_f1[i] = ctc_f
        #     tr_precision[i] = tr_p
        #     tr_recall[i] = tr_r
        #     tr_f1[i] = tr_f
        # #print(ctc_precision)
        # self.test_ctc_token_precision(ctc_precision.mean())
        # self.test_ctc_token_recall(ctc_recall.mean())
        # self.test_ctc_token_f1(ctc_f1.mean())
        # self.test_tr_token_precision(tr_precision.mean())
        # self.test_tr_token_recall(tr_recall.mean())
        # self.test_tr_token_f1(tr_f1.mean())
        # # self.test_ctc_ter(self.ter(ctc_preds_tokens, gt_tokens))
        # # self.test_tr_ter(self.ter(tr_preds_tokens, gt_tokens))

        # self.log(
        #     "test/ctc_token_p",
        #     self.test_ctc_token_precision,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "test/ctc_token_r",
        #     self.test_ctc_token_recall,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "test/ctc_token_f",
        #     self.test_ctc_token_f1,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        # self.log(
        #     "test/token_tr_p",
        #     self.test_tr_token_precision,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "test/token_tr_r",
        #     self.test_tr_token_recall,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "test/token_tr_f",
        #     self.test_tr_token_f1,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # # self.log(
        # #     "test/ctc_ter",
        # #     self.test_ctc_ter,
        # #     on_step=False,
        # #     on_epoch=True,
        # #     prog_bar=True,
        # # )
        # # self.log(
        # #     "test/tr_ter", self.test_tr_ter, on_step=False, on_epoch=True, prog_bar=True
        # # )

    def configure_optimizers(self):
        """Setting the optimizer for training."""
        optimizer = self.hparams.optimizer(
            filter(lambda p: p.requires_grad, self.parameters())
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


    # def tatum_to_pos_dur(self, tab):
    #     pos_dur_pitch = []
    #     for i, note in enumerate(tab):
    #         pos, onset, pitch = note
    #         if i == 0:
    #             start_index = i
    #             start_pos = pos
    #             current_pitch = pitch
    #         if onset == 1 and i != 0:
    #             dur = i - start_index
    #             pos_dur_pitch.append([start_pos, dur, current_pitch])
    #             start_index = i
    #             if dur == 1:
    #                 start_index -= 1
    #             start_pos = pos
    #             current_pitch = pitch
    #     dur = i - start_index + 1
    #     pos_dur_pitch.append([start_pos, dur, current_pitch])
    #     return pos_dur_pitch
    
    # def tatum_to_pos_dur(self, tab):
    #     pos_dur_pitch = []
    #     start_index = 0
    #     start_pos = 0
    #     current_pitch = 0
    #     current_string = 0
    #     current_fret = 0
    #     for i, note in enumerate(tab):
    #         #pos, onset, pitch, string, fret = note
    #         pos, onset, pitch, string, fret, tech = note
    #         if i == 0:
    #             start_index = i
    #             start_pos = pos
    #             current_pitch = pitch
    #             current_string = string
    #             current_fret = fret
                
    #             current_tech = tech
                
    #         if onset == 1 and i != 0:
    #             dur = i - start_index
    #             pos_dur_pitch.append([start_pos, dur, current_pitch, current_string, current_fret])
    #             start_index = i
    #             if dur == 1:
    #                 start_index -= 1
    #             start_pos = pos
    #             current_pitch = pitch
    #             current_string = string
    #             current_fret = fret
                
    #             current_tech = tech

    #     dur = len(tab) - start_index
    #     pos_dur_pitch.append([start_pos, dur, current_pitch, current_string, current_fret])
    #     return pos_dur_pitch
    
    # def make_pianoroll(self, tab, tempo):
    #     """Make pianoroll from MIDI

    #     Returns:
    #         pianoroll: pianoroll representation of MIDI
    #     """
    #     pm = pretty_midi.PrettyMIDI(initial_tempo=80)
    #     inst = pretty_midi.Instrument(program=30, is_drum=False, name='my guitar')
    #     # pm.instruments.append(inst)

    #     velocity = 100
    #     tatum_time = 5 / tempo
    #     bar = 0
    #     last_pos = -1
    #     grace = False
    #     for pos, dur, pitch, string, fret in tab:
    #         if pos <= last_pos:
    #             bar += 1
    #             last_pos = pos
    #         else:
    #             last_pos = pos
    #         if grace:
    #             dur -= 1
    #         start = (bar * 48 + pos) * tatum_time
    #         end = start + dur * tatum_time
    #         inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
    #         if dur == 1:
    #             grace = True
    #         else:
    #             grace = False
    #     pm.instruments.append(inst)
    #     pianoroll = torch.from_numpy(
    #         pm.get_piano_roll().T
    #     ).to(self.device)
    #     pianoroll = (pianoroll > 0).int()
    #     return pianoroll

    # def make_pianoroll(self, tab, tempo):
    #     """Make pianoroll from MIDI

    #     Returns:
    #         pianoroll: pianoroll representation of MIDI
    #     """
    #     pm = pretty_midi.PrettyMIDI(initial_tempo=80)
    #     inst = pretty_midi.Instrument(program=30, is_drum=False, name='my guitar')
    #     # pm.instruments.append(inst)

    #     velocity = 100
    #     tatum_time = 5 / tempo
    #     bar = 0
    #     last_pos = -1
    #     grace = False
    #     for pos, dur, pitch in tab:
    #         if pos <= last_pos:
    #             bar += 1
    #             last_pos = pos
    #         else:
    #             last_pos = pos
    #         if grace:
    #             dur -= 1
    #         start = (bar * 48 + pos) * tatum_time
    #         end = start + dur * tatum_time
    #         inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
    #         if dur == 1:
    #             grace = True
    #         else:
    #             grace = False
    #     pm.instruments.append(inst)
    #     pianoroll = torch.from_numpy(
    #         pm.get_piano_roll().T
    #     ).to(self.device)
    #     pianoroll = (pianoroll > 0).int()
    #     return pianoroll

    # def ter(self, pred, gt):
    #     """Calculate token error rate (TER).

    #     Args:
    #         pred: Argmaxed prediction.
    #         gt: The ground truth.

    #     Returns:
    #         Token error rate.
    #     """
    #     pred_sentences = []
    #     gt_sentences = []

    #     def make_sentence(tokens):
    #         # tokens -> sentence
    #         sentence = " ".join([str(int(word)) for word in tokens])
    #         return [sentence]

    #     for i in range(len(pred)):
    #         pred_sentences = pred_sentences + make_sentence(pred[i])
    #         gt_sentences = gt_sentences + make_sentence(gt[i])

    #     ter = word_error_rate(preds=pred_sentences, target=gt_sentences)

    #     return ter

    # def get_precision_recall_f1(self, pred, gt):
    #     """Calculate precision, recall and F1 score between the prediction and the ground truth label.

    #     Args:
    #         pred: Argmaxed prediction.
    #         gt: The ground truth.

    #     Returns:
    #         precision, recall and F1 score.
    #     """
    #     # TP = 0.5
    #     # FP = 0.5
    #     # FN = 0.5
    #     TP = (gt * pred).sum()
    #     FP = ((1 - gt) * pred).sum()
    #     FN = (gt * (1 - pred)).sum()
    #     precision = TP / (TP + FP + 1e-7)
    #     recall = TP / (TP + FN + 1e-7)
    #     f1 = 2 * precision * recall / (precision + recall + 1e-7)
    #     return precision, recall, f1

    # def plot_output(
    #     self, gt_pianoroll, ctc_preds_pianoroll, tr_preds_pianoroll, track_name
    # ):
    #     """Function for plotting the predictions.

    #     Args:
    #         gt_pianoroll: Pianoroll representation of the ground truth.
    #         ctc_preds_pianoroll: Pianoroll representation of the prediction from the encoder.
    #         tr_preds_pianoroll: Pianoroll representation of the prediction from the Transformer decoder.
    #         track_name: Name of the track. Used to set the filename.
    #     """
    #     figs_path = os.path.join(self.figs_output_dir, track_name + ".png")
    #     TP_patch = mpatches.Patch(color="white", label="TP")
    #     FP_patch = mpatches.Patch(color="yellow", label="FP")
    #     FN_patch = mpatches.Patch(color="red", label="FN")

    #     plt.subplot(2, 1, 1)
    #     plt.title("CTC")
    #     fused = (gt_pianoroll * 0.4 + ctc_preds_pianoroll).cpu()
    #     sns.heatmap(
    #         fused[:, 52:101].T, cmap="hot", cbar=False, rasterized=False
    #     ).invert_yaxis()
    #     plt.legend(handles=[TP_patch, FP_patch, FN_patch], loc="upper right")
    #     plt.tick_params(
    #         left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    #     )
    #     for n_note in range(4 * 16):
    #         if n_note % 16 == 0:
    #             plt.axvline(len(gt_pianoroll) * n_note / 64, color="white", lw=1)
    #         else:
    #             plt.axvline(len(gt_pianoroll) * n_note / 64, color="white", lw=0.2)

    #     plt.subplot(2, 1, 2)
    #     plt.title("Transformer")
    #     fused = (gt_pianoroll * 0.4 + tr_preds_pianoroll).cpu()
    #     sns.heatmap(
    #         fused[:, 52:101].T, cmap="hot", cbar=False, rasterized=False
    #     ).invert_yaxis()
    #     plt.legend(handles=[TP_patch, FP_patch, FN_patch], loc="upper right")
    #     plt.tick_params(
    #         left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    #     )
    #     for n_note in range(4 * 16):
    #         if n_note % 16 == 0:
    #             plt.axvline(len(gt_pianoroll) * n_note / 64, color="white", lw=1)
    #         elif n_note % 4 == 0:
    #             plt.axvline(len(gt_pianoroll) * n_note / 64, color="white", lw=0.4)
    #         else:
    #             plt.axvline(len(gt_pianoroll) * n_note / 64, color="white", lw=0.1)
    #     plt.tight_layout()
    #     plt.savefig(figs_path, dpi=400)
    #     plt.close("all")

    # def plot_attention_map(self, track_name_list):
    #     """Function for plotting the attention map of self-attention and cross-attention.

    #     Args:
    #         track_name_list: List of track names.
    #     """
    #     n_layer = self.network.transformer_decoder.n_layer
    #     n_head = self.network.transformer_decoder.n_head

    #     plot_counter = 1
    #     for layer_n in range(n_layer):
    #         attention_map = (
    #             self.network.transformer_decoder.transformer_decoder.decoders._modules[
    #                 f"{layer_n}"
    #             ]
    #             ._modules["src_attn"]
    #             .attn.cpu()
    #         )
    #         #print(attention_map.shape)
    #         for head_n in range(n_head):
    #             plt.subplot(n_head, n_layer, plot_counter, aspect="equal")
    #             cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    #             plt.imshow(
    #                 attention_map[0, head_n],
    #                 cmap=cmap,
    #                 norm=LogNorm(vmin=1e-3),
    #                 aspect="auto",
    #             )
    #             plt.gcf().set_size_inches(12, 12)
    #             plt.axis("off")
    #             plt.tick_params(
    #                 left=False,
    #                 right=False,
    #                 labelleft=False,
    #                 labelbottom=False,
    #                 bottom=False,
    #             )
    #             plt.tight_layout()
    #             plot_counter += 1
    #     plt.savefig(
    #         f"{self.figs_output_dir}/epoch_{self.current_epoch}_{track_name_list[0]}_src_attn.png",
    #         dpi=200,
    #     )
    #     plt.close("all")


if __name__ == "__main__":
    import hydra
    import omegaconf
    #import pyrootutils
    from src.tokenizer.tokenizer import TabTokenizer, tokenizer_initializer

    # root = pyrootutils.setup_root(__file__, pythonpath=True)
    root = os.path.abspath(os.curdir)
    cfg_tokenizer = omegaconf.OmegaConf.load(os.path.join(root, "conf" , "tokenizer" , "tokenizer.yaml"))
    initializer = tokenizer_initializer(
        pitch_range = cfg_tokenizer.pitch_range,
        pos_range = cfg_tokenizer.pos_range,
        dur_range = cfg_tokenizer.dur_range,
        mask = cfg_tokenizer.mask,
        sos_eos = cfg_tokenizer.sos_eos,
    )
    tokenizer, vocab, vocab_size = initializer.generate()
    cfg = omegaconf.OmegaConf.load(os.path.join(root, "conf" , "model" , "lit_hybrid_ctc.yaml"))
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "lit_hybrid_ctc.yaml")
    _ = hydra.utils.instantiate(cfg, tokenizer=tokenizer, vocab=vocab, vocab_size=vocab_size)
