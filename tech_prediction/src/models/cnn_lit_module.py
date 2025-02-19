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
    """LightningModule for automatic guitar transcription using cnn model."""

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

        self.network = cnn_net()

        # loss function
        self.loss_func = loss_func

        self.output_dir = output_dir

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        self.test_frame_pitch_f1s = MeanMetric()
        self.test_frame_string_fret_f1s = MeanMetric()
        self.test_frame_tech_f1s = MeanMetric()

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
        group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits = self.forward(
            batch["padded_cqt"],
            batch["cqt_lens"],
            batch["padded_target_gt"],
            batch["target_lens_gt"],
            batch["frame_level_note_attribs"],
            batch["note_attribs"],
        )
        loss = self.loss_func(
            group_logits,
            tech_logits,
            final_tech_logits,
            batch["cqt_lens"],
            batch["padded_target_gt"].long(),
            batch["target_lens_gt"],
        )
        return loss, group_logits, tech_logits, final_tech_logits, frame_level_final_tech_logits

    def model_inference(self, padded_cqt, cqt_lens, frame_level_note_attribs, note_attribs):
        group_preds, tech_preds, final_tech_preds, frame_level_final_tech_preds = self.network.inference(
            padded_cqt, cqt_lens, frame_level_note_attribs, note_attribs
        )
        return group_preds, tech_preds, final_tech_preds, frame_level_final_tech_preds

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}


    def on_test_epoch_start(self):
        self.tech_output_dir = os.path.join(self.output_dir, "tech")
        self.gt_tab_dir = os.path.join(self.output_dir, "gt_tab")
        self.gt_frame_level_dir = os.path.join(self.output_dir, "gt_frame_level")
        self.full_tech_prediction_output_dir = os.path.join(self.output_dir, "full_tech_prediction")
        self.frame_level_full_tech_prediction_output_dir = os.path.join(self.output_dir, "frame_level_full_tech_prediction")
        if not os.path.exists(self.tech_output_dir):
            os.makedirs(self.tech_output_dir)
        if not os.path.exists(self.gt_tab_dir):
            os.makedirs(self.gt_tab_dir)
        if not os.path.exists(self.gt_frame_level_dir):
            os.makedirs(self.gt_frame_level_dir)
        if not os.path.exists(self.full_tech_prediction_output_dir):
            os.makedirs(self.full_tech_prediction_output_dir)
        if not os.path.exists(self.frame_level_full_tech_prediction_output_dir):
            os.makedirs(self.frame_level_full_tech_prediction_output_dir)

    def test_step(self, batch, batch_idx):
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
            final_tech_prediction = [int(element) for row in final_tech_prediction for element in row]
            pitch_string_fret = [row[2:] for row in note_attribs]
            final_tech_prediction_note_level = self.tatum_to_note(final_tech_prediction, dur, pitch_string_fret)
            frame_level_final_tech_prediction = [frame_level_final_tech_preds[i]]
            frame_level_final_tech_prediction = [int(element) for row in frame_level_final_tech_prediction for element in row]
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

            gt_final_tech = [gt_target[i]]
            gt_final_tech = [int(element) for row in gt_final_tech for element in row]
            gt_final_tech_note_level = [[row[0], row[1], row[-1]] for row in self.tatum_to_note(gt_final_tech, dur, pitch_string_fret)]
            gt_final_tech_path = os.path.join(
                self.tech_output_dir, track_name_list[i] + "_gt.tsv"
            )
            with open(gt_final_tech_path, 'w') as f:
                for line in gt_final_tech:
                    f.write(f"{line}\n")
            f.close

            gt_full_final_tech = [full_target[i]]
            gt_full_final_tech = np.array(gt_full_final_tech, dtype=int)
            gt_full_final_tech = gt_full_final_tech.squeeze()
            gt_tab_path = os.path.join(
                self.gt_tab_dir, track_name_list[i] + "_gt_tab.tsv"
            )
            np.savetxt(gt_tab_path, gt_full_final_tech, fmt='%d')

            gt_frame_level_full_final_tech = [frame_level_gt[i]]
            gt_frame_level_full_final_tech = np.array(gt_frame_level_full_final_tech, dtype=int)
            gt_frame_level_full_final_tech = gt_frame_level_full_final_tech.squeeze()
            gt_frame_level_tab_path = os.path.join(
                self.gt_frame_level_dir, track_name_list[i] + "_gt_frame_level_tab.tsv"
            )
            np.savetxt(gt_frame_level_tab_path, gt_frame_level_full_final_tech, fmt='%d')

            self.calculate_frame_level_score(gt_frame_level_full_final_tech, frame_level_full_attribs, frame_level_note_attribs, frame_tp_list, frame_fp_list, frame_fn_list, frame_final_tech_tp_list, frame_final_tech_fp_list, frame_final_tech_fn_list)
        
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
    
    def tatum_to_frame(self, tatum_tab, frame_len):
        indices = np.linspace(0, len(tatum_tab)-1, frame_len)
        indices = np.round(indices).astype(int)
        upsampled_tab = [tatum_tab[i] for i in indices]
        return upsampled_tab


    def tatum_to_note(self, tech, duration, pitch_string_fret):
        start = 0
        notes = []
        for dur in duration:
            if start >= len(tech):
                break
            sublist = tech[start:start+dur]
            count = Counter(sublist)
            majority = count.most_common(1)[0][0]
            sublist = pitch_string_fret[start:start+dur]
            pitch = [row[0] for row in sublist]
            count = Counter(pitch)
            majority_pitch = count.most_common(1)[0][0]
            for row in sublist:
                if row[0] == majority_pitch:
                    majority_pitch = row[0]
                    majority_string = row[1]
                    majority_fret = row[2]
                    break
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

    def calculate_frame_level_score(self, gt, pred, attrib_pred, tp_list, fp_list, fn_list, final_tech_tp_list, final_tech_fp_list, final_tech_fn_list):
        pred = np.array(pred)
        attrib_pred = np.array(attrib_pred)
        onset_times = np.arange(pred.shape[0])
        pred = np.hstack((onset_times[:, None], pred))
        onset_times = np.arange(gt.shape[0])
        gt = np.hstack((onset_times[:, None], gt))

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

        all_gt_indices = set(range(len(gt)))
        matched_gt_indices = {gt_idx for gt_idx, _ in tech_matching}
        unmatched_gt_indices = all_gt_indices - matched_gt_indices
        for gt_idx in unmatched_gt_indices:
            final_tech_fn_list[gt[gt_idx, -1] - 1] += 1

        all_pred_indices = set(range(len(pred)))
        matched_pred_indices = {pred_idx for _, pred_idx in tech_matching}
        unmatched_pred_indices = all_pred_indices - matched_pred_indices
        for pred_idx in unmatched_pred_indices:
            final_tech_fp_list[pred[pred_idx, -1] - 1] += 1
