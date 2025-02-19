from typing import Any, List, Optional
import torch
import os
import sys
sys.path.append(os.getcwd())
from matplotlib import pyplot as plt
import librosa
import pretty_midi
import librosa.display
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import seaborn as sns
from pytorch_lightning import LightningModule
from torchmetrics.functional import word_error_rate
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from collections import Counter
import numpy as np
import itertools
import mir_eval
from statistics import mean


class CNNLitModule(LightningModule):
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
        cnn_net: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: any,
        output_dir: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['loss_func'])

        # pass vocab size to initialize the partially initialized network
        self.network = cnn_net()

        # loss function
        self.loss_func = loss_func

        # dir for saving the results (midi)
        self.output_dir = output_dir

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # metrics container
        # self.val_loss = MeanMetric()
        # self.val_ctc_ter = MeanMetric()

        self.test_frame_pitch_f1s = MeanMetric()
        self.test_frame_string_fret_f1s = MeanMetric()
        self.test_frame_tech_f1s = MeanMetric()
        # self.test_ctc_token_precision = MeanMetric()
        # self.test_ctc_token_recall = MeanMetric()
        # self.test_ctc_token_f1 = MeanMetric()
        # self.test_tech_f1 = MeanMetric()
        # self.test_group_f1 = MeanMetric()
        # self.test_ctc_ter = MeanMetric()
        # self.error_rate = MeanMetric()

    def forward(
        self,
        padded_cqt,
        cqt_lens,
        padded_target_gt,
        target_lens_gt,
        frame_level_note_attribs,
        note_attribs,
    ):
        return self.network(
            padded_cqt, frame_level_note_attribs, note_attribs
        )

    def model_step(self, batch: dict):
        #ctc_preds_logprob = self.forward(
        group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits = self.forward(
            batch["padded_cqt"],
            batch["cqt_lens"],
            batch["padded_target_gt"],
            batch["target_lens_gt"],
            batch["frame_level_note_attribs"],
            batch["note_attribs"],
        )
        #print(final_tech_logits.shape)
        #print(ctc_preds_logprob.shape)
        loss = self.loss_func(
            #ctc_preds_logprob,
            group_logits,
            tech_logits,
            final_tech_logits,
            batch["cqt_lens"],
            batch["padded_target_gt"].long(),
            batch["target_lens_gt"],
        )

        #return loss, ctc_preds_logprob
        return loss, group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits

    def model_inference(self, padded_cqt, cqt_lens, frame_level_note_attribs, note_attribs):
        #ctc_preds_tokens = self.network.inference(
        group_preds, tech_preds, final_tech_preds, frame_level_final_tech_preds = self.network.inference(
            padded_cqt, cqt_lens, frame_level_note_attribs, note_attribs
        )
        #return ctc_preds_tokens
        return group_preds, tech_preds, final_tech_preds, frame_level_final_tech_preds

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    #     #loss, ctc_preds_logprob = self.model_step(batch)
    #     loss, group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits = self.model_step(batch)

    #     # update and log metrics
    #     self.val_loss(loss)
    #     self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    #     # ctc tokens
    #     # preds_ctc_tokens = self.network.ctc_decoder(
    #     #     ctc_preds_logprob, batch["cqt_lens"]
    #     # )
    #     final_tech_preds = torch.argmax(final_tech_logits, dim=-1)

    #     gt_tokens = [
    #         batch["padded_tokens_gt"][i, : batch["token_lens_gt"][i]].tolist()
    #         for i in range(len(batch["token_lens_gt"]))
    #     ]

    #     #self.val_ctc_ter(self.ter(preds_ctc_tokens, gt_tokens))
    #     self.val_ctc_ter(self.ter(final_tech_preds, gt_tokens))
    #     self.log(
    #         "val/ctc_ter", self.val_ctc_ter, on_step=False, on_epoch=True, prog_bar=True
    #     )

    #     return {"loss": loss}

    def on_test_epoch_start(self):
        self.tech_output_dir = os.path.join(self.output_dir, "tech")
        self.gt_tab_dir = os.path.join(self.output_dir, "gt_tab")
        self.gt_frame_level_dir = os.path.join(self.output_dir, "gt_frame_level")
        # self.figs_output_dir = os.path.join(self.output_dir, "figs")
        # self.score_output_dir = os.path.join(self.output_dir, "score")
        self.full_tech_prediction_output_dir = os.path.join(self.output_dir, "full_tech_prediction")
        # self.tech_prediction_output_dir = os.path.join(self.output_dir, "tech_prediction")
        # self.group_prediction_output_dir = os.path.join(self.output_dir, "group_prediction")
        self.frame_level_full_tech_prediction_output_dir = os.path.join(self.output_dir, "frame_level_full_tech_prediction")
        if not os.path.exists(self.tech_output_dir):
            os.makedirs(self.tech_output_dir)
        if not os.path.exists(self.gt_tab_dir):
            os.makedirs(self.gt_tab_dir)
        if not os.path.exists(self.gt_frame_level_dir):
            os.makedirs(self.gt_frame_level_dir)
        # if not os.path.exists(self.figs_output_dir):
        #     os.makedirs(self.figs_output_dir)
        # if not os.path.exists(self.score_output_dir):
        #     os.makedirs(self.score_output_dir)
        if not os.path.exists(self.full_tech_prediction_output_dir):
            os.makedirs(self.full_tech_prediction_output_dir)
        # if not os.path.exists(self.tech_prediction_output_dir):
        #     os.makedirs(self.tech_prediction_output_dir)
        # if not os.path.exists(self.group_prediction_output_dir):
        #     os.makedirs(self.group_prediction_output_dir)
        if not os.path.exists(self.frame_level_full_tech_prediction_output_dir):
            os.makedirs(self.frame_level_full_tech_prediction_output_dir)

    def test_step(self, batch, batch_idx):
        #ctc_preds_tokens = self.model_inference(
        group_preds, tech_preds, final_tech_preds, frame_level_final_tech_preds = self.model_inference(
            batch["padded_cqt"],
            batch["cqt_lens"],
            batch["frame_level_note_attribs"],
            batch["note_attribs"],
        )

        padded_target_gt = batch["padded_target_gt"]
        target_lens_gt = batch["target_lens_gt"]
        track_name_list = batch["track_name_list"]
        full_target = batch["full_target"]
        frame_level_gt = batch["frame_level_gt"]
        cqt_lens = batch["cqt_lens"]
        full_target_lens_gt = batch["full_target_lens_gt"]

        # plot attention map
        #self.plot_attention_map(track_name_list)

        gt_target = [
            padded_target_gt[i, : target_lens_gt[i]].tolist()
            for i in range(len(padded_target_gt))
        ]

        full_target = [
            full_target[i, : full_target_lens_gt[i]].tolist()
            for i in range(len(full_target))
        ]

        frame_level_gt = [
            frame_level_gt[i, : cqt_lens[i]].tolist()
            for i in range(len(frame_level_gt))
        ]

        frame_tp_list = [0, 0, 0]
        frame_fn_list = [0, 0, 0]
        frame_fp_list = [0, 0, 0]

        frame_final_tech_tp_list = [0] * 24
        frame_final_tech_fn_list = [0] * 24
        frame_final_tech_fp_list = [0] * 24

        # ctc_precision = torch.zeros(len(ctc_preds_tokens))
        # ctc_recall = torch.zeros(len(ctc_preds_tokens))
        # ctc_f1 = torch.zeros(len(ctc_preds_tokens))

        # tech_precision = torch.zeros(len(tech_preds))
        # tech_recall = torch.zeros(len(tech_preds))
        # tech_f1 = torch.zeros(len(tech_preds))

        # group_precision = torch.zeros(len(group_preds))
        # group_recall = torch.zeros(len(group_preds))
        # group_f1 = torch.zeros(len(group_preds))

        # ctc_preds_note_level_techs = []
        # gt_note_level_techs = []

        # group_dict = {
        #     0: [0],
        #     1: [1],
        #     2: [2, 3, 4],
        #     3: [5, 6, 7],
        #     4: [8, 9]
        # }
        # tech_dict = {
        #     0: [0],
        #     1: [1, 19],
        #     2: [2, 3, 4],
        #     3: [5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 24],
        #     4: [13],
        #     5: [17],
        #     6: [15],
        #     7: [18],
        #     8: [16],
        #     9: [14, 23]
        # }
        # inverted_tech_dict = {value: key for key, values in tech_dict.items() for value in values}
        # inverted_group_dict = {value: key for key, values in group_dict.items() for value in values}

        for i in range(len(final_tech_preds)):
            tempo = batch["tempos"][i].item()
            frame_level_note_attribs = batch["frame_level_note_attribs"][i].tolist()
            frame_level_note_attribs = [[int(element) for element in row] for row in frame_level_note_attribs]
            note_attribs = batch["note_attribs"][i].tolist()
            note_attribs = [[int(element) for element in row] for row in note_attribs]
            dur = []
            count = 0
            for _, onset, _, _, _ in note_attribs:
                if onset == 1 and count != 0:
                    dur.append(count)
                    count = 0
                count += 1
            dur.append(count)

            final_tech_prediction = [final_tech_preds[i]]
            #print(ctc_preds_tech)
            final_tech_prediction = [int(element) for row in final_tech_prediction for element in row]
            # tech_prediction = [tech_preds[i]]
            # tech_prediction = [int(element) for row in tech_prediction for element in row]
            # group_prediction = [group_preds[i]]
            # group_prediction = [int(element) for row in group_prediction for element in row]
            #ctc_preds_tech = [int(element) for element in ctc_preds_tech]
            pitch_string_fret = [row[2:] for row in note_attribs]
            #frame_level_final_tech_gt = self.tatum_to_frame()
            final_tech_prediction_note_level = self.tatum_to_note(final_tech_prediction, dur, pitch_string_fret)
            # ctc_preds_tech_note_level = [[row[0], row[1], row[-1]] for row in final_tech_prediction_note_level]

            # tech_prediction_note_level = self.tatum_to_note(tech_prediction, dur, pitch_string_fret)
            # tech_note_level = [[row[0], row[1], row[-1]] for row in tech_prediction_note_level]

            # group_prediction_note_level = self.tatum_to_note(group_prediction, dur, pitch_string_fret)
            # group_note_level = [[row[0], row[1], row[-1]] for row in group_prediction_note_level]

            # frame_level_final_tech_prediction_path = os.path.join(
            #     self.frame_level_full_tech_prediction_output_dir, track_name_list[i] + "_frame_level_final_tech_full_prediction.tsv"
            # )
            frame_level_final_tech_prediction = [frame_level_final_tech_preds[i]]
            frame_level_final_tech_prediction = [int(element) for row in frame_level_final_tech_prediction for element in row]
            # with open(frame_level_final_tech_prediction_path, 'w') as f:
            #     for line in frame_level_final_tech_prediction:
            #         f.write(f"{line}\n")
            # f.close

            frame_level_final_tech_prediction_path = os.path.join(
                self.frame_level_full_tech_prediction_output_dir, track_name_list[i] + "_frame_level_note_attribs.tsv"
            )
            frame_level_full_attribs = [sublist + [element] for sublist, element in zip(frame_level_note_attribs, frame_level_final_tech_prediction)]
            with open(frame_level_final_tech_prediction_path, 'w') as f:
                for line in frame_level_full_attribs:
                    f.write(f"{line}\n")
            f.close

            full_tech_prediction_path = os.path.join(
                self.full_tech_prediction_output_dir, track_name_list[i] + "_final_tech_full_prediction.tsv"
            )
            with open(full_tech_prediction_path, 'w') as f:
                for line in final_tech_prediction_note_level:
                    f.write(f"{line}\n")
            f.close

            # tech_prediction_path = os.path.join(
            #     self.tech_prediction_output_dir, track_name_list[i] + "_tech_full_prediction.tsv"
            # )
            # with open(tech_prediction_path, 'w') as f:
            #     for line in tech_prediction_note_level:
            #         f.write(f"{line}\n")
            # f.close

            # group_prediction_path = os.path.join(
            #     self.group_prediction_output_dir, track_name_list[i] + "_group_full_prediction.tsv"
            # )
            # with open(group_prediction_path, 'w') as f:
            #     for line in group_prediction_note_level:
            #         f.write(f"{line}\n")
            # f.close


            # ctc_preds_tatum_tech_path = os.path.join(
            #     self.tech_output_dir, track_name_list[i] + "_ctc_pred_tatum_level.tsv"
            # )
            # with open(ctc_preds_tatum_tech_path, 'w') as f:
            #     for line in ctc_preds_tech:
            #         f.write(f"{line}\n")
            # f.close

            # ctc_preds_tech_path = os.path.join(
            #     self.tech_output_dir, track_name_list[i] + "_ctc_pred.tsv"
            # )
            # with open(ctc_preds_tech_path, 'w') as f:
            #     for line in ctc_preds_tech_note_level:
            #         f.write(f"{line}\n")
            # f.close

            # ctc_preds_note_level_techs.append([row[-1] for row in ctc_preds_tech_note_level])

            gt_final_tech = [gt_target[i]]
            gt_final_tech = [int(element) for row in gt_final_tech for element in row]
            #gt_tech = [int(element) for element in gt_tech]
            gt_final_tech_note_level = [[row[0], row[1], row[-1]] for row in self.tatum_to_note(gt_final_tech, dur, pitch_string_fret)]
            gt_final_tech_path = os.path.join(
                self.tech_output_dir, track_name_list[i] + "_gt.tsv"
            )
            with open(gt_final_tech_path, 'w') as f:
                #for line in gt_final_tech_note_level:
                for line in gt_final_tech:
                    f.write(f"{line}\n")
            f.close

            gt_full_final_tech = [full_target[i]]
            #gt_full_final_tech = [int(element) for row in gt_full_final_tech for element in row]
            #gt_full_final_tech = [[int(element) for element in row] for row in gt_full_final_tech]
            gt_full_final_tech = np.array(gt_full_final_tech, dtype=int)
            gt_full_final_tech = gt_full_final_tech.squeeze()
            gt_tab_path = os.path.join(
                self.gt_tab_dir, track_name_list[i] + "_gt_tab.tsv"
            )
            np.savetxt(gt_tab_path, gt_full_final_tech, fmt='%d')
            # with open(gt_tab_path, 'w') as f:
            #     for line in gt_full_final_tech:
            #         f.write(f"{line}\n")
            # f.close

            gt_frame_level_full_final_tech = [frame_level_gt[i]]
            gt_frame_level_full_final_tech = np.array(gt_frame_level_full_final_tech, dtype=int)
            gt_frame_level_full_final_tech = gt_frame_level_full_final_tech.squeeze()
            #print(gt_frame_level_full_final_tech_numpy.shape)
            #gt_frame_level_full_final_tech = [int(element) for row in gt_frame_level_full_final_tech for element in row]
            gt_frame_level_tab_path = os.path.join(
                self.gt_frame_level_dir, track_name_list[i] + "_gt_frame_level_tab.tsv"
            )
            np.savetxt(gt_frame_level_tab_path, gt_frame_level_full_final_tech, fmt='%d')

            self.calculate_frame_level_score(gt_frame_level_full_final_tech, frame_level_full_attribs, frame_level_note_attribs, frame_tp_list, frame_fp_list, frame_fn_list, frame_final_tech_tp_list, frame_final_tech_fp_list, frame_final_tech_fn_list)

            # with open(gt_frame_level_tab_path, 'w') as f:
            #     for line in gt_frame_level_full_final_tech:
            #         f.write(f"{line}\n")
            # f.close

        #     gt_note_level_techs.append([row[-1] for row in gt_final_tech_note_level])

        #     ctc_preds_pianoroll = self.make_pianoroll(ctc_preds_tech_note_level, tempo)
        #     gt_pianoroll = self.make_pianoroll(gt_final_tech_note_level, tempo)

        #     if len(ctc_preds_pianoroll) > len(gt_pianoroll):
        #         ctc_preds_pianoroll = ctc_preds_pianoroll[: len(gt_pianoroll)]
        #     elif len(ctc_preds_pianoroll) < len(gt_pianoroll):
        #         zeros = torch.zeros_like(gt_pianoroll)
        #         zeros[: len(ctc_preds_pianoroll)] = ctc_preds_pianoroll
        #         ctc_preds_pianoroll = zeros
        #     self.plot_output(
        #         gt_pianoroll,
        #         ctc_preds_pianoroll,
        #         track_name_list[i],
        #     )

        #     ctc_p, ctc_r, ctc_f = self.get_precision_recall_f1(
        #         ctc_preds_pianoroll, gt_pianoroll
        #     )

        #     gt_tech_note_level = [row[:-1] + [inverted_tech_dict[row[-1]]] for row in gt_final_tech_note_level]
        #     tech_pianoroll = self.make_pianoroll(tech_note_level, tempo)
        #     gt_tech_pianoroll = self.make_pianoroll(gt_tech_note_level, tempo)

        #     if len(tech_pianoroll) > len(gt_tech_pianoroll):
        #         tech_pianoroll = tech_pianoroll[: len(gt_tech_pianoroll)]
        #     elif len(tech_pianoroll) < len(gt_tech_pianoroll):
        #         zeros = torch.zeros_like(gt_tech_pianoroll)
        #         zeros[: len(tech_pianoroll)] = tech_pianoroll
        #         tech_pianoroll = zeros
        #     tech_p, tech_r, tech_f = self.get_precision_recall_f1(
        #         tech_pianoroll, gt_tech_pianoroll
        #     )

        #     gt_group_note_level = [row[:-1] + [inverted_group_dict[row[-1]]] for row in gt_tech_note_level]
        #     group_pianoroll = self.make_pianoroll(group_note_level, tempo)
        #     gt_group_pianoroll = self.make_pianoroll(gt_group_note_level, tempo)

        #     if len(group_pianoroll) > len(gt_group_pianoroll):
        #         group_pianoroll = group_pianoroll[: len(gt_group_pianoroll)]
        #     elif len(group_pianoroll) < len(gt_group_pianoroll):
        #         zeros = torch.zeros_like(gt_group_pianoroll)
        #         zeros[: len(group_pianoroll)] = group_pianoroll
        #         group_pianoroll = zeros
        #     group_p, group_r, group_f = self.get_precision_recall_f1(
        #         group_pianoroll, gt_group_pianoroll
        #     )
        #     # each_tech_precision[i] = ctc_p[:-1]
        #     # each_tech_recall[i] = ctc_r[:-1]
        #     # each_tech_f1[i] = ctc_f[:-1]
        #     #print(each_tech_precision)

        #     score_path = os.path.join(
        #         self.score_output_dir, track_name_list[i] + ".tsv"
        #     )
        #     with open(score_path, 'w') as f:
        #         for elem in ctc_p[:-1]:
        #             f.write(f"{elem.item()} ")
        #         f.write("\n")
        #         for elem in ctc_r[:-1]:
        #             f.write(f"{elem.item()} ")
        #         f.write("\n")
        #         for elem in ctc_f[:-1]:
        #             f.write(f"{elem.item()} ")
        #         f.write("\n")
        #     f.close

        #     ctc_precision[i] = ctc_p[-1]
        #     ctc_recall[i] = ctc_r[-1]
        #     ctc_f1[i] = ctc_f[-1]

        #     tech_precision[i] = tech_p[-1]
        #     tech_recall[i] = tech_r[-1]
        #     tech_f1[i] = tech_f[-1]

        #     group_precision[i] = group_p[-1]
        #     group_recall[i] = group_r[-1]
        #     group_f1[i] = group_f[-1]
        
        frame_f1s = torch.tensor([self.calculate_f1(tp, fp, fn) for tp, fp, fn in zip(frame_tp_list, frame_fp_list, frame_fn_list)])
        self.test_frame_pitch_f1s(frame_f1s[0].mean())
        self.log(
            "frame_pitch_f1s",
            self.test_frame_pitch_f1s,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.test_frame_string_fret_f1s(frame_f1s[1].mean())
        self.log(
            "frame_string_fret_f1s",
            self.test_frame_string_fret_f1s,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.test_frame_tech_f1s(frame_f1s[2].mean())
        self.log(
            "frame_tech_f1s",
            self.test_frame_tech_f1s,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


        frame_final_tech_f1s = torch.tensor([self.calculate_f1(tp, fp, fn) for tp, fp, fn in zip(frame_final_tech_tp_list, frame_final_tech_fp_list, frame_final_tech_fn_list)])

        #print(final_tech_f1s)
        
        # # each_tech_precision = torch.tensor([[elem.item() for elem in list] for list in each_tech_precision])
        # # each_tech_recall = torch.tensor([[elem.item() for elem in list] for list in each_tech_recall])
        # # each_tech_f1 = torch.tensor([[elem.item() for elem in list] for list in each_tech_f1])
        # # # print(each_tech_precision)
        # # # print(each_tech_recall)
        # # # print(each_tech_f1)
        # # precision_mean = self.calculate_mean(each_tech_precision)
        # # recall_mean = self.calculate_mean(each_tech_recall)
        # # f1_mean = self.calculate_mean(each_tech_f1)

        # # score_path = os.path.join(
        # #     self.score_output_dir, "frame_level_.tsv"
        # # )
        # # with open(score_path, 'w') as f:
        # #     f.write(f"precision: {precision_mean}\n")
        # #     f.write(f"recall: {recall_mean}\n")
        # #     f.write(f"f1: {f1_mean}\n")
        # # f.close


        # self.test_ctc_token_precision(ctc_precision.mean())
        # self.test_ctc_token_recall(ctc_recall.mean())
        # self.test_ctc_token_f1(ctc_f1.mean())
        # self.test_tech_f1(tech_f1.mean())
        # self.test_group_f1(group_f1.mean())

        # #self.test_ctc_ter(self.ter(ctc_preds_tokens, gt_tokens))
        # self.test_ctc_ter(self.ter(ctc_preds_note_level_techs, gt_note_level_techs))

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
        #     "test/tech_f",
        #     self.test_tech_f1,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "test/group_f",
        #     self.test_group_f1,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        # self.log(
        #     "test/error_rate",
        #     self.test_ctc_ter,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        # self.error_rate(self.ter_without_normal(ctc_preds_note_level_techs, gt_note_level_techs))

        # self.log(
        #     "test/error_rate_without_normal",
        #     self.error_rate,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

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
    #     for pos, dur, tech in tab:
    #         if pos <= last_pos:
    #             bar += 1
    #             last_pos = pos
    #         else:
    #             last_pos = pos
    #         if grace:
    #             dur -= 1
    #         start = (bar * 48 + pos) * tatum_time
    #         end = start + dur * tatum_time
    #         inst.notes.append(pretty_midi.Note(velocity, tech+60, start, end))
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
    
    def tatum_to_frame(self, tatum_tab, frame_len):
        indices = np.linspace(0, len(tatum_tab)-1, frame_len)
        indices = np.round(indices).astype(int)
        upsampled_tab = [tatum_tab[i] for i in indices]
        return upsampled_tab


    def tatum_to_note(self, tech, duration, pitch_string_fret):
        #print(tech)
        #print(duration)
        start = 0
        notes = []
        for dur in duration:
            if start >= len(tech):
                break
            sublist = tech[start:start+dur]
            #print(sublist)
            count = Counter(sublist)
            #print(count.most_common(1))
            majority = count.most_common(1)[0][0]
            sublist = pitch_string_fret[start:start+dur]
            #print(sublist)
            pitch = [row[0] for row in sublist]
            count = Counter(pitch)
            majority_pitch = count.most_common(1)[0][0]
            for row in sublist:
                if row[0] == majority_pitch:
                    majority_pitch = row[0]
                    majority_string = row[1]
                    majority_fret = row[2]
                    break
            # majority_row = [row for row in sublist if row[0] == majority_pitch]
            # print(majority_row)
            notes.append([start % 48, dur, majority_pitch, majority_string, majority_fret, majority])
            start += dur
        return notes
    
    def calculate_tp_fn_fp(self, gt, pred, indices, matching):
        tp = 0
        fn = 0
        fp = 0
        
        for gt_idx, pred_idx in matching:
            gt_value = tuple(gt[gt_idx, idx] for idx in indices)
            pred_value = tuple(pred[pred_idx, idx] for idx in indices)

            if gt_value == pred_value:
                tp += 1
            else:
                fn += 1
                fp += 1
        matched_gt_indices = {gt_idx for gt_idx, _ in matching}
        fn += len(gt) - len(matched_gt_indices)

        matched_pred_indices = {pred_idx for _, pred_idx in matching}
        fp += len(pred) - len(matched_pred_indices)

        return tp, fn, fp

    def calculate_f1(self, tp, fn, fp):
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        if tp == 0 and fn == 0 and fp == 0:
            return -1
        return f1

    # gt_path = "D:/solo_tech/data/preprocessed/frame_level_gt"
    # #pred_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-09-17_21-27-08/frame_level_full_tech_prediction"
    # pred_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-09-19_09-21-15/frame_level_full_tech_prediction"
    # # considered in loss function
    # pred_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-09-25_13-23-48/frame_level_full_tech_prediction"
    # prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_02-19-33/frame_level_full_tech_prediction"

    # attrib_pred_path = "D:/solo_tech/data/preprocessed/frame_level_pred"

    # file_name = glob.glob(os.path.join(gt_path, "*npy"))
    #file_name = glob.glob(os.path.join(prediction_path, "*tsv"))
    

    # files = []
    # for file in file_name:
    #     idx = (os.path.split(file)[1][:-31])
    #     files.append(os.path.join("D:/solo_tech/data/preprocessed/full_target/" + idx + ".npy"))
    # print(files)
    # file_name = glob.glob(os.path.join(prediction_path, "*tsv"))
    # files = []
    # for file in file_name:
    #     idx = (os.path.split(file)[1][:-29])
    #     files.append(os.path.join("D:/solo_tech/data/preprocessed/frame_level_gt/" + idx + ".npy"))
    def calculate_frame_level_score(self, gt, pred, attrib_pred, tp_list, fp_list, fn_list, final_tech_tp_list, final_tech_fp_list, final_tech_fn_list):
        pred = np.array(pred)
        attrib_pred = np.array(attrib_pred)
        #attrib_pred = np.load(os.path.join(attrib_pred_path, idx + ".npy"))
        #attrib_pred = pred[:, :3]
        #print(idx)
        #print(pred)
        #print(attrib_pred)
        onset_times = np.arange(pred.shape[0])
        #print(onset_times)

        pred = np.hstack((onset_times[:, None], pred))
        #print(pred)
        onset_times = np.arange(gt.shape[0])
        gt = np.hstack((onset_times[:, None], gt))
        #print(gt)

        onset_times = np.arange(attrib_pred.shape[0])
        attrib_pred = np.hstack((onset_times[:, None], attrib_pred))

        new_column = (gt[:, 0] + 1)/ 100
        reference_onsets = np.column_stack((gt[:, 0] / 100, new_column))
        adjusted_midi = [0 if midi == 100 else 101 if midi == 101 else midi - 12 for midi in gt[:, 1]]
        reference_pitches = np.array([mir_eval.util.midi_to_hz(midi) for midi in adjusted_midi])
        reference_tech = gt[:, -1]

        new_column = (pred[:, 0] + 1) / 100
        estimated_tech_onsets = np.column_stack((pred[:, 0] / 100, new_column))
        estimated_tech = pred[:, -1]

        new_column = (attrib_pred[:, 0] + 1)/ 100
        estimated_onsets = np.column_stack((attrib_pred[:, 0] / 100, new_column))
        adjusted_midi = [0 if midi == 100 else 101 if midi == 101 else midi - 12 for midi in attrib_pred[:, 1]]
        estimated_pitches = np.array([mir_eval.util.midi_to_hz(midi) for midi in adjusted_midi])

        onset_matching = mir_eval.transcription.match_notes(
            reference_onsets,
            reference_pitches,
            estimated_onsets,
            estimated_pitches,
            onset_tolerance=0.03,
            #pitch_tolerance=0,
            offset_ratio=None
        )

        tech_matching = mir_eval.transcription.match_notes(
            reference_onsets,
            reference_tech,
            estimated_tech_onsets,
            estimated_tech,
            onset_tolerance=0.03,
            pitch_tolerance=0,
            offset_ratio=None
        )
        #print(reference_tech)
        #print(estimated_tech)

        tp_list[0] += len(onset_matching)
        fp_list[0] += len(attrib_pred) - len(onset_matching)
        fn_list[0] += len(gt) - len(onset_matching)

        tp, fn, fp = self.calculate_tp_fn_fp(gt, attrib_pred, [1, 2], onset_matching)
        tp_list[1] += tp
        fp_list[1] += fp
        fn_list[1] += fn

        tp_list[2] += len(tech_matching)
        fp_list[2] += len(pred) - len(tech_matching)
        fn_list[2] += len(gt) - len(tech_matching)

        for gt_idx, pred_idx in tech_matching:
            final_tech_tp_list[gt[gt_idx, -1] - 1] += 1
            # group_tp_list[inverted_group_dict[inverted_tech_dict[gt[gt_idx, -1]]] - 1] += 1


        all_gt_indices = set(range(len(gt)))
        matched_gt_indices = {gt_idx for gt_idx, _ in tech_matching}
        unmatched_gt_indices = all_gt_indices - matched_gt_indices
        for gt_idx in unmatched_gt_indices:
            final_tech_fn_list[gt[gt_idx, -1] - 1] += 1
            # group_fn_list[inverted_group_dict[inverted_tech_dict[gt[gt_idx, -1]]] - 1] += 1

        all_pred_indices = set(range(len(pred)))
        matched_pred_indices = {pred_idx for _, pred_idx in tech_matching}
        unmatched_pred_indices = all_pred_indices - matched_pred_indices
        for pred_idx in unmatched_pred_indices:
            final_tech_fp_list[pred[pred_idx, -1] - 1] += 1
            # group_fp_list[inverted_group_dict[inverted_tech_dict[pred[pred_idx, -1]]] - 1] += 1

        #break

        # for i in range(len(gt)):
        #     #if gt[i][0] == pred[i][0]:
        #     if gt[i][1] == attrib_pred[i][0]:
        #         tp_list[0] += 1
        #     else:
        #         fn_list[0] += 1
        #         fp_list[0] += 1
        #     if gt[i][-1] == pred[i][-1]:
        #         tp_list[-1] += 1
        #     else:
        #         fn_list[-1] += 1
        #         fp_list[-1] += 1
        #     if gt[i][2] == attrib_pred[i][1] and gt[i][3] == attrib_pred[i][2]:
        #         tp_list[1] += 1
        #     else:
        #         fn_list[1] += 1
        #         fp_list[1] += 1

        # for i in range(len(gt)):
        #     if inverted_tech_dict[gt[i][-1]] == inverted_tech_dict[pred[i][-1]]:
        #         tech_tp_list[inverted_tech_dict[gt[i][-1]] - 1] += 1
        #     else:
        #         tech_fn_list[inverted_tech_dict[gt[i][-1]] - 1] += 1
        #         tech_fp_list[inverted_tech_dict[pred[i][-1]] - 1] += 1
        #break
    # print(tech_tp_list)
    # print(tech_fn_list)
    # print(tech_fp_list)
    # print(tp_list)
    # print(fn_list)
    

    '''-------------------------------------------------------------------------------------------'''
        
    # def calculate_mean(self, input_tensor):
    #     mask = input_tensor != -1
    #     masked_tensor = input_tensor * mask
    #     count_valid = mask.sum(dim=0)
    #     sum_valid = masked_tensor.sum(dim=0)
    #     mean = sum_valid/count_valid
    #     return mean
        

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
    #     #print(pred_sentences)
    #     ter = word_error_rate(preds=pred_sentences, target=gt_sentences)

    #     return ter
    
    # def ter_without_normal(self, pred, gt):
    #     mismatches = 0
    #     length = 0
    #     valid_pos = 0
    #     for i in range(len(pred)):
    #         for el1, el2 in zip(pred[i], gt[i]):
    #             if el2 != 19:
    #                 valid_pos += 1
    #                 if el1 != el2:
    #                     mismatches += 1
    #         # mismatches += sum(el1 != el2 for el1, el2 in zip(pred[i], gt[i]))
    #         # length += len(pred[i])
    #     error_rate = mismatches / valid_pos if valid_pos > 0 else 0
    #     return error_rate
    
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
    #     #print(gt)
    #     precisions = []
    #     recalls = []
    #     f1s = []
    #     for i in range(61, 85):
    #         TP = (gt[:, i] * pred[:, i]).sum()
    #         FP = ((1 - gt[:, i]) * pred[:, i]).sum()
    #         FN = (gt[:, i] * (1 - pred[:, i])).sum()
    #         if TP == 0 and FP == 0 and FN == 0:
    #             precisions.append(torch.tensor(-1))
    #             recalls.append(torch.tensor(-1))
    #             f1s.append(torch.tensor(-1))
    #         else:
    #             precision = TP / (TP + FP + 1e-7)
    #             recall = TP / (TP + FN + 1e-7)
    #             f1 = 2 * precision * recall / (precision + recall + 1e-7)
    #             precisions.append(precision)
    #             recalls.append(recall)
    #             f1s.append(f1)
    #     TP = (gt * pred).sum()
    #     FP = ((1 - gt) * pred).sum()
    #     FN = (gt * (1 - pred)).sum()
    #     precision = TP / (TP + FP + 1e-7)
    #     recall = TP / (TP + FN + 1e-7)
    #     f1 = 2 * precision * recall / (precision + recall + 1e-7)
    #     precisions.append(precision)
    #     recalls.append(recall)
    #     f1s.append(f1)
    #     #print(TP, FP, FN)
    #     return precisions, recalls, f1s

    # def plot_output(
    #     self, gt_pianoroll, ctc_preds_pianoroll, track_name
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

    #     #plt.subplot(2, 1, 1)
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

    #     plt.tight_layout()
    #     plt.savefig(figs_path, dpi=400)
    #     plt.close("all")