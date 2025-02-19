import torch
import torch.nn.functional as F
import glob
import os
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# import sys
# sys.path.append(os.getcwd())
# from src.tokenizer.tokenizer import TabTokenizer, tokenizer_initializer
# from src.data_preprocess.data_preprocess import solo_preprocess
from tokenizer.tokenizer import TabTokenizer, tokenizer_initializer
from data_preprocess.data_preprocess import solo_preprocess
from shutil import rmtree
from rich.progress import Progress
from multiprocessing import Pool
from itertools import repeat
import hydra
from omegaconf import DictConfig, OmegaConf
import random


class PadCollate:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, batch):
        cqt_max_len = 0
        token_max_len = 0
        for _, cqt, token, _ in batch:
            if cqt_max_len < cqt.shape[0]:
                cqt_max_len = cqt.shape[0]
            if token_max_len < token.shape[0]:
                token_max_len = token.shape[0]

        padded_cqt = torch.zeros((len(batch), cqt_max_len, batch[0][1].shape[1]))
        padded_token = torch.zeros((len(batch), token_max_len), dtype=torch.long)
        cqt_lens = torch.zeros((len(batch)), dtype=torch.long)
        token_lens = torch.zeros((len(batch)), dtype=torch.long)
        tempos = torch.zeros((len(batch)))
        track_name_list = []
        for i, (track_name, cqt, token, tempo) in enumerate(batch):
            track_name_list += [track_name]
            padded_cqt[i, : len(cqt)] = cqt
            padded_token[i, : len(token)] = token
            cqt_lens[i] = len(cqt)
            token_lens[i] = len(token)
            tempos[i] = tempo

        out = {
            "track_name_list": track_name_list,
            "padded_cqt": padded_cqt,
            "cqt_lens": cqt_lens,
            "tempos": tempos,
            "padded_tokens_gt": padded_token,
            "token_lens_gt": token_lens,
        }
        return out

class SoloDataset(Dataset):
    def __init__(self, track_name_list, data_dir, tokenizer):
        self.track_name_list = track_name_list
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.vocab = tokenizer._create_vocabulary()
        self.eos_idx = self.vocab.__getitem__("EOS_None")
        self.cqt_dir = os.path.join(self.data_dir, "cqt")
        self.target_dir = os.path.join(self.data_dir, "target")
        self.tempo_dir = os.path.join(self.data_dir, "tempo")


    def __len__(self):
        return len(self.track_name_list)

    def __getitem__(self, idx):
        cqt_path = os.path.join(self.cqt_dir, f"{self.track_name_list[idx]}.npy")
        target_path = os.path.join(self.target_dir, f"{self.track_name_list[idx]}.npy")
        tempo_path = os.path.join(self.tempo_dir, f"{self.track_name_list[idx]}.npy")
        cqt = np.load(cqt_path)
        target = np.load(target_path)
        tempo = np.load(tempo_path)
        tokens = self.tokenizer.tab_to_tokens(target)
        tokens += [self.eos_idx]

        sample = (
            self.track_name_list[idx],
            torch.from_numpy(cqt),
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(tempo, dtype=torch.long),
        )

        return sample

class SoloDatasetModule(LightningDataModule):
    def __init__(
        self,
        data_preprocess_cfg: any,
        tokenizer: TabTokenizer,
        vocab_size: int,
        data_dir: str = "data/",
        train_test_split_ratio = 0.9,
        batch_size: int = 16,
        num_workers: int = 10,
        dataloader_workers: int = 5,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        preprocess_on_training_start: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.audio_filename_list = glob.glob(
            os.path.join(
                data_preprocess_cfg.data_dir, "wav", "*.wav"
            )
        )

    def prepare_data(self):
        """Make dirs, preprocess and split the dataset"""
        output_dir = self.hparams.data_preprocess_cfg.output_dir
        cqt_dir = os.path.join(output_dir, "cqt")
        target_dir = os.path.join(output_dir, "target")
        tempo_dir = os.path.join(output_dir, "tempo")

        if self.hparams.preprocess_on_training_start:
            # make dirs for data preprocessing
            if os.path.exists(output_dir):
                rmtree(output_dir)
                os.makedirs(cqt_dir)
                os.makedirs(target_dir)
                os.makedirs(tempo_dir)
            else:
                os.makedirs(cqt_dir)
                os.makedirs(target_dir)
                os.makedirs(tempo_dir)

            # preprocess the data using multiprocessing for parallel computation
            pool = Pool(processes=self.hparams.num_workers)
            with Progress() as p:
                task = p.add_task(
                    "Preprocessing...", total=len(self.audio_filename_list)
                )
                for _ in pool.imap_unordered(
                    solo_preprocess,
                    zip(
                        self.audio_filename_list,
                        repeat(self.hparams.data_preprocess_cfg),
                    ),
                ):
                    p.update(task, advance=1)

            pool.close()

        # split the dataset to train/test set
        split_trackname_list = glob.glob(os.path.join(cqt_dir, "*.npy"))
        split_trackname_list = [
            os.path.split(path)[1][:-4] for path in split_trackname_list
        ]
        random.shuffle(split_trackname_list)
        
        # self.train_data_list = split_trackname_list[
        #     : int(round(len(split_trackname_list) * self.hparams.train_test_split_ratio))
        # ]
        # self.val_data_list = []
        # self.test_data_list = split_trackname_list[
        #     int(round(len(split_trackname_list) * self.hparams.train_test_split_ratio)) :
        # ]

        # random result
        self.train_data_list = ['20_9_02', 'gym_13_014', '50_17_04', '20_2_03', '20_19_08', '20_7_015', '20_7_021', '20_16_05', '50_37_06', '50_2_04', '50_31_00', '20_4_05', '20_13_02', '50_37_029', '50_51_011', '20_12_04', '20_16_018', '50_47_012', '50_51_012', '20_1_010', '20_16_09', '50_2_00', '20_18_08', '50_22_015', '20_11_09', '50_2_02', '20_8_00', '50_8_06', '20_19_05', 'gym_9_00', '50_10_04', 'gym_13_013', '20_15_05', '50_36_010', '50_29_011', '50_15_01', '50_2_010', '50_44_08', '50_22_012', '50_44_01', '50_20_014', 'gym_12_00', '20_19_014', '20_1_02', '50_38_03', 'gym_12_02', '50_21_06', '20_6_011', '20_3_014', '50_24_010', '20_16_08', '50_28_02', '20_7_024', '50_51_013', '50_8_09', 'gym_5_06', '20_4_014', '20_16_017', '20_12_06', '20_8_014', '50_18_00', '20_5_04', '50_38_05', '20_19_07', '20_15_04', '50_22_05', '50_51_00', 'gym_13_010', '50_1_04', '50_1_010', '50_42_02', '50_19_03', '50_43_016', '20_11_017', '20_3_010', '20_1_08', '50_23_05', '20_5_010', '50_20_06', '20_11_019', 'gym_13_011', '50_20_021', '20_8_018', '20_6_019', '20_7_019', '20_12_01', '50_4_05', '20_17_010', 'gym_19_011', '20_16_014', '20_11_021', '50_26_01', '50_29_09', '50_24_02', '20_2_09', '50_24_01', '50_26_02', '50_38_04', '50_47_04', '20_3_01', 'gym_19_00', '20_2_017', 'gym_12_01', '20_11_022', '50_4_04', '20_15_01', '50_23_07', '20_8_09', '20_2_020', '50_44_07', '50_21_03', '50_32_01', 'gym_22_08', '20_17_00', '50_44_09', '50_37_028', '20_5_00', '50_23_00', '20_16_010', '20_11_018', '20_4_013', '20_19_019', '20_19_013', '50_43_05', '20_4_03', '50_46_02', '50_37_022', '20_19_023', '50_21_010', 'gym_19_015', '20_19_016', '20_19_01', '50_13_04', 'gym_22_01', '50_46_04', '20_13_00', '50_31_05', '20_11_020', '20_12_05', '50_49_07', '50_19_01', '20_18_011', '20_17_013', '20_8_03', '20_4_08', '20_13_03', '50_22_01', 'gym_5_00', '50_36_015', '20_19_02', 'gym_13_02', '20_15_08', '50_51_019', 'gym_22_05', '20_19_022', '20_3_012', '50_36_017', 'gym_13_01', '50_16_01', '20_1_09', '20_1_012', '20_6_01', '50_43_014', '50_16_02', '20_18_01', '20_7_03', '50_8_04', '50_20_019', '20_6_03', '50_20_04', '50_37_010', '50_37_019', '20_19_03', '50_38_00', '50_15_00', '20_16_012', '50_44_04', '50_26_08', '50_5_02', '50_28_00', 'gym_5_05', '50_48_013', '20_4_06', '50_24_09', '50_37_026', '50_37_020', '50_39_04', '20_17_09', '20_11_011', '50_40_08', '50_24_011', '50_6_02', '20_11_015', '50_2_01', '20_2_00', '50_14_011', 'gym_12_04', '50_26_06', '20_18_02', '50_26_03', '20_19_017', '50_37_012', '50_20_08', '50_22_07', 'gym_22_010', '50_43_011', '50_1_09', '20_11_04', 'gym_19_013', 'gym_12_03', '50_26_07', '20_14_04', '50_17_08', '20_3_00', '50_42_010', '50_36_06', '20_2_021', '20_5_09', '20_3_05', 'gym_19_03', '50_9_02', '50_4_00', '20_1_06', '50_51_014', '20_18_09', '20_19_018', '20_16_011', '50_11_03', '50_39_00', '50_26_09', '20_2_013', '50_37_08', '50_15_02', '50_20_011', '50_20_07', '50_17_09', '20_5_013', '20_6_012', '20_4_09', '50_50_04', '20_1_00', 'gym_5_08', '50_47_02', '50_33_00', '20_7_020', '50_14_03', '50_22_013', '50_47_05', '20_14_06', '20_11_00', '50_37_025', '20_17_05', '50_40_03', '50_51_02', '50_13_06', 'gym_22_07', '50_48_014', '50_14_08', '50_21_09', '20_2_01', '50_30_06', '50_6_00', 'gym_19_01', '20_8_04', '50_2_011', '50_9_01', '50_43_09', '20_13_06', '50_28_04', '20_7_018', '20_2_014', '50_42_03', '50_20_03', '20_19_012', '20_16_022', '20_17_08', 'gym_19_014', '50_31_06', '20_8_017', 'gym_13_07', 'gym_13_06', '50_24_07', '50_17_07', '50_43_024', '50_5_04', '50_18_04', '20_10_00', '20_5_05', '20_6_017', '50_49_06', '50_25_01', '50_36_012', '50_47_010', '50_22_011', '20_18_010', '50_13_05', 'gym_19_04', '50_12_00', '50_45_00', 'gym_13_09', '50_21_02', '50_46_00', 'gym_9_03', '50_4_03', '20_1_020', '50_20_09', '50_43_017', '20_3_02', '20_12_08', '20_7_01', '50_48_09', '20_5_08', '50_14_01', '50_37_013', '50_48_011', '50_51_09', '50_44_010', '20_7_02', '50_8_01', '50_36_05', 'gym_19_05', '50_11_04', '50_21_01', '50_29_012', '50_30_09', '20_6_09', '50_21_013', 'gym_9_04', '50_43_012', '50_44_02', '50_23_03', '50_3_04', '50_44_05', '20_7_016', '50_8_02', '20_11_08', '20_4_02', '20_3_09', '20_8_020', '20_17_03', '50_1_07', '50_51_07', '50_48_012', '20_1_07', '20_7_09', '50_50_00', '50_43_015', '20_17_07', '50_20_020', '50_1_03', '50_44_011', '50_10_00', 'gym_19_06', '20_17_014', '50_31_03', '50_22_017', '50_30_05', '50_8_07', '20_4_00', '50_2_08', '50_7_04', '20_6_013', '50_49_02', '20_7_00', '20_2_018', '50_29_01', '50_43_02', '50_26_010', '20_16_00', '20_18_012', '50_20_01', '20_19_024', '50_29_010', 'gym_19_017', '20_7_08', '50_2_07', '50_49_05', '50_43_027', '50_7_02', '20_2_016', '50_30_012', '20_1_014', '50_48_07', '50_21_08', '50_20_023', '50_5_01', '20_7_05', '50_31_02', '50_23_08', 'gym_5_02', '20_19_021', '50_20_05', '20_17_02', '50_19_04', '50_51_05', '20_8_07', '50_22_019', '20_3_08', '50_30_011', '20_8_05', '20_15_02', 'gym_5_07', 'gym_19_012', '50_49_03', '20_6_015', '20_1_019', '50_14_010', '50_43_019', '20_14_01', '50_22_04', '50_29_08', '20_11_014', '50_48_015', '50_51_021', '50_39_02', '50_36_013', '50_1_05', '20_7_028', '50_10_02', '20_7_010', '50_35_00', '50_31_08', '50_50_02', '20_15_010', '50_36_02', '20_3_013', '50_3_02', '50_1_06', '50_33_01', '50_48_02', '50_42_08', '50_24_08', '50_17_03', '50_21_07', '50_49_08', '50_20_017', '50_37_09', '20_13_011', '50_25_03', '20_16_024', '50_43_04', '20_11_01', '50_8_00', '50_13_01', '50_14_07', '50_37_07', '50_44_00', '20_19_09', 'gym_22_011', '50_22_08', '50_43_01', '20_3_015', '50_14_05', '50_29_02', '50_37_024', '20_1_013', '20_19_011', '50_44_06', '50_14_04', '20_2_06', '50_37_03', '50_38_01', '50_12_02', '20_16_06', '50_51_020', '50_22_03', '20_6_014', '50_37_030', '50_22_06', '50_40_07', '20_5_06', '50_44_03', 'gym_19_016', '50_51_024', '50_32_02', 'gym_19_07', '50_8_011', '50_50_01', '20_11_06', '20_7_027', '50_48_016', '50_51_022', '50_15_04', '50_14_00', '50_42_07', '20_17_015', '50_40_05', '20_6_07', '50_37_00', '20_7_023', '20_12_02', '20_8_06', '50_49_04', '50_22_02', '50_51_023', '50_42_01', '50_36_07', '50_28_01', '50_36_08', '20_7_04', '50_37_04', '20_15_00', '20_2_012', '50_5_00', '50_47_06', '50_5_06', '50_43_020', '50_42_06', '50_43_07', '50_43_06', '50_20_018', '20_16_019', '20_2_04', '50_3_00', '50_36_00', '50_47_09', '50_1_01', '20_16_023', '50_42_05', '50_43_013', '50_7_01', '20_13_09', 'gym_13_08', '20_18_00', '20_12_00', '50_4_01', '20_4_010', '50_30_07', '50_43_023', '50_1_012', '20_18_05', '50_43_026', '50_48_010', '50_40_04', '20_11_012', '20_5_011', '20_15_06', '20_5_03', '20_7_012', '50_12_03', '50_21_04', '20_16_07', '20_15_011', '50_8_010', '50_26_00', '20_7_017', '20_15_012', '20_19_020', '50_47_013', '50_43_022', '50_30_03', '20_8_011', '50_36_020', '50_25_05', '50_6_01', '50_22_010', '50_51_08', '20_18_06', '50_44_012', '50_2_012', '20_16_020', '50_43_010', '20_6_018', 'gym_5_04', '20_11_02', '50_42_04', '50_31_07', '50_43_03', '20_8_08', '50_43_018', '50_36_011', '20_19_04', '50_21_05', '50_20_010', '50_20_013', '50_21_011', '50_24_03', '20_3_011', 'gym_19_02', '50_42_00', '20_1_04', '20_17_01', '50_22_018', '20_19_015', '50_48_01', '20_9_01', '50_47_07', '50_32_00', '20_17_011', '50_11_02', '50_12_04', '20_1_017', 'gym_22_03', '20_16_016', '20_9_04', '20_15_09', '20_1_015', '50_39_03', 'gym_22_06', '20_8_013', '20_12_07', '20_13_08', '50_46_01', '20_15_03', '20_1_011', '50_48_04', '20_8_02', '20_3_07', '50_23_06', '50_47_01', '50_37_02', '50_48_00', '20_19_00', '50_51_06', '50_19_00', '50_2_06', '20_14_05', '50_49_09', '20_3_03', '50_37_014', '50_22_09', '20_1_018', '50_32_04', '20_15_07', '20_6_08', '50_9_00', 'gym_5_01', '20_7_025', '50_20_022', '50_26_012', 'gym_13_00', '50_1_02', '50_40_01', '50_36_016', '50_49_01', 'gym_13_012', '20_2_02', '50_51_01', '50_11_00', '50_8_05', '20_7_022', '50_4_02', '50_5_05', '20_8_012', '20_7_011', '20_8_01', '50_24_04', '20_6_016', '50_22_014', '20_13_07', '50_47_011', 'gym_5_03', '20_5_01', '50_24_06', '50_20_015', '50_6_04', '50_51_03', '50_7_00', '50_17_01', '50_22_00', '20_3_04', '20_8_015', 'gym_9_02', '50_48_06', '50_29_07', '50_37_021', '20_14_02', '20_1_03', '50_36_03', '20_8_019', '50_51_018', '50_36_09', 'gym_13_03', '50_16_03', '50_29_04', '20_3_016', '20_18_03', '20_1_05', '50_14_06', '20_18_07', '20_4_01', '20_6_05', '20_13_04', '50_26_011', '50_5_03', '50_25_00', '20_11_05', '50_14_09', '50_1_00', '50_22_016', '50_47_08', '50_3_03', '50_1_08', '50_30_00', '20_6_02', '20_11_013', '20_11_016', 'gym_19_09', '50_18_03', '20_18_013', 'gym_22_04', '50_25_04', '50_20_024', '50_37_018', '50_38_06', '50_23_04', '50_31_04', '50_21_00', '50_24_00', 'gym_9_01', '50_2_09', '20_6_06', '50_30_010', '20_7_07', '50_30_08', '20_16_013', '50_17_02', '20_16_01', '20_6_010', '20_19_010', '20_17_012', '20_17_06', '50_13_00', 'gym_22_012', '50_29_00', '50_18_02', '20_16_021', '50_26_05', '50_51_017', '20_16_015', '20_11_07', '50_30_01', '50_28_03', 'gym_22_00', '50_20_00', '50_12_01', '50_17_06', '50_1_011', '50_23_02', '50_37_015', '50_36_018', '20_8_010', '50_40_00', '50_3_01', 'gym_13_04', '50_37_023', '20_16_03', '50_5_07', '50_29_06', '20_13_01', '20_6_00', '50_16_00', '20_1_01', '20_12_03', '50_14_02', '50_18_01', '20_5_07', '50_26_013', '50_11_01', '50_2_03', '50_20_012', '20_13_05', '20_4_011', '50_49_00', 'gym_19_08', '50_13_03', '50_42_09', '50_46_03', '50_37_016', '50_36_04', '20_7_013', '20_3_06', '20_5_02', '20_13_010', '20_15_013', '50_31_01', '50_8_03']
        self.val_data_list = []
        self.test_data_list = ['20_7_014', '50_8_08', '50_37_01', '20_18_04', '20_9_03', '50_51_010', '20_2_015', '20_2_010', 'gym_22_09', '50_43_025', '50_51_04', '50_43_021', '20_19_06', '50_36_014', '50_30_04', '20_10_01', '50_21_012', '50_37_011', '50_48_08', '50_48_03', '50_10_03', '50_37_017', '50_47_00', '50_20_016', '20_11_010', '50_20_02', '50_8_012', '50_48_05', '20_2_05', '50_32_03', '50_43_00', '50_7_03', '20_17_04', '20_4_07', '20_5_012', '50_40_06', '50_47_03', '20_17_016', '50_25_02', '20_7_026', '20_1_016', 'gym_22_02', '50_50_03', '20_9_00', '50_51_016', '20_14_03', '50_19_02', '50_39_01', '20_16_02', '50_10_01', '50_14_012', '50_14_013', '20_16_04', '50_15_03', '20_11_03', '20_2_07', '50_38_02', '50_30_02', '50_36_019', '20_1_021', '20_2_019', '50_37_05', '50_17_05', '50_6_03', '50_51_015', '50_37_027', '50_24_05', '20_2_08', '50_2_05', '50_17_00', '50_23_01', '20_7_06', '20_4_04', '50_26_04', '50_36_01', '50_13_02', '50_29_03', '20_8_016', '50_43_08', '50_29_05', '50_40_02', '20_4_012', '20_14_00', 'gym_19_010', '20_2_011', '20_6_04', 'gym_13_05']


    def setup(self, stage=None):
        """Initialize train, validation and test dataset"""
        self.data_train = SoloDataset(
            self.train_data_list,
            self.hparams.data_preprocess_cfg.output_dir,
            self.hparams.tokenizer,
        )
        # self.data_val = SoloDataset(
        #     self.val_data_list,
        #     self.hparams.data_preprocess_cfg.output_dir,
        #     self.hparams.tokenizer,
        # )
        self.data_test = SoloDataset(
            self.test_data_list,
            self.hparams.data_preprocess_cfg.output_dir,
            self.hparams.tokenizer,
        )

    def train_dataloader(self):
        """Initialize the train dataloader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=PadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    # def val_dataloader(self):
    #     """Initialize the validataion dataloader"""
    #     # if len(self.data_val) == 0:
    #     #     return None
    #     return DataLoader(
    #         dataset=self.data_val,
    #         batch_size=self.hparams.batch_size,
    #         collate_fn=PadCollate(self.hparams.vocab_size),
    #         num_workers=self.hparams.dataloader_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=False,
    #     )

    def test_dataloader(self):
        """Initialize the test dataloader"""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=PadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


@hydra.main(version_base='1.3', config_path='../../conf/dataset', config_name='dataset.yaml')
def main(cfg: DictConfig) -> None:
    root = os.path.abspath(os.curdir)
    cfg_preprocess = OmegaConf.load(os.path.join(root, "conf" , "data_preprocess" , "data_preprocess.yaml"))
    cfg_tokenizer = OmegaConf.load(os.path.join(root, "conf" , "tokenizer" , "tokenizer.yaml"))
    initializer = tokenizer_initializer(
        pitch_range = cfg_tokenizer.pitch_range,
        pos_range = cfg_tokenizer.pos_range,
        dur_range = cfg_tokenizer.dur_range,
        mask = cfg_tokenizer.mask,
        sos_eos = cfg_tokenizer.sos_eos,
    )
    tokenizer, vocab, vocab_size = initializer.generate()
    datamodule = SoloDatasetModule(data_preprocess_cfg=cfg_preprocess, tokenizer=tokenizer, vocab_size=vocab_size)
    datamodule.prepare_data()

if __name__ == "__main__":
    main()
    

