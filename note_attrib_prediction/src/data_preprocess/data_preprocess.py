import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
import glob
from omegaconf import DictConfig, OmegaConf
from itertools import repeat


def z_score_normalize(x: any) -> any:
    mean, std = x.mean(), x.std()
    return (x-mean)/std

def process_cqt(
    audio,
    tempo,
    original_sr,
    track_name,
    split_unit_in_bars,
    split_hop_bar_len,
    hparams,
    output_dir,
):
    len_in_sec = librosa.get_duration(y=audio, sr=original_sr)
    len_in_bars = int(round((tempo * len_in_sec) / (4 * 60)))
    n_units = int(
        round((tempo * len_in_sec) / (split_hop_bar_len * split_unit_in_bars * 60))
        - ((split_unit_in_bars - split_hop_bar_len) // split_hop_bar_len)
    )
    audio = audio.astype(float)
    audio = librosa.resample(
        audio, orig_sr=original_sr, target_sr=hparams["down_sampling_rate"]
    )
    if hparams["normalize_wave"]:
        audio = librosa.util.normalize(audio)
    cqt = librosa.cqt(
        audio,
        hop_length=hparams["hop_length"],
        sr=hparams["down_sampling_rate"],
        n_bins=hparams["total_n_bins"],
        bins_per_octave=hparams["bins_per_octave"],
    )
    if hparams["db_scale"]:
        cqt = librosa.amplitude_to_db(np.abs(cqt))
    if hparams["normalize_cqt"]:
        cqt = z_score_normalize(cqt)
    cqt = np.abs(cqt).T

    for unit_n in range(n_units):
        cqt_filename = os.path.join(
            output_dir, "cqt", f"{track_name}_0{unit_n}.npy"
        )
        st = int(round(len(cqt) * unit_n * split_hop_bar_len / len_in_bars))
        end = (
            int(
                round(len(cqt))
                * ((unit_n * split_hop_bar_len + split_unit_in_bars) / len_in_bars)
            )
            if int(
                round(len(cqt))
                * ((unit_n * split_hop_bar_len + split_unit_in_bars) / len_in_bars)
            )
            <= len(cqt)
            else len(cqt)
        )
        np.save(cqt_filename, cqt[st:end])


def get_tempo(
    track_name,
    song_dur,
    label_dir,
    output_dir
):
    label_filename = os.path.join(label_dir, f"{track_name}.tsv")
    with open(label_filename, "r") as f:
        lines = [line.split() for line in f.read().split("\n")][:-1]
    f.close()
    label = [list(map(int, line)) for line in lines]
    tempo = 60 * (label[-1][0] + ((label[-1][1] + label[-1][2]) / 48)) * 4 / song_dur
    return tempo


def get_target_file(
    track_name,
    label_dir,
    output_dir,
    prediction_dir,
):
    label_filename = os.path.join(label_dir, f"{track_name}.tsv")
    with open(label_filename, "r") as f:
        lines = [line.split() for line in f.read().split("\n")][:-1]
    f.close()
    label = [list(map(int, line)) for line in lines]
    filelist = glob.glob(os.path.join(output_dir, "cqt", f"{track_name}_*.*"))
    
    for file in filelist:
        idx = int((os.path.split(file)[1][:-4]).split("_")[-1])
        target_filename = os.path.join(
            output_dir, "target", f"{track_name}_0{idx}.npy"
        )
        prediction_filename = os.path.join(
            prediction_dir, f"{track_name}_0{idx}.npy"
        )
        bar_st = idx
        bar_end = bar_st + 3
        executed = False
        for i in range(len(label)):
            if label[i][0] == bar_st and executed == False:
                st = i
                executed = True
            if label[i][0] == bar_end:
                end = i
        target = label[st:end+1]
        target = np.array(target)
        target = np.delete(target, 0, 1)
        np.save(target_filename, target)
        np.save(prediction_filename, target)


def get_tempo_file(
    track_name,
    tempo,
    output_dir
):
    filelist = glob.glob(os.path.join(output_dir, "cqt", f"{track_name}_*.*"))
    for file in filelist:
        idx = int((os.path.split(file)[1][:-4]).split("_")[-1])
        target_filename = os.path.join(
            output_dir, "tempo", f"{track_name}_0{idx}.npy"
        )
        np.save(target_filename, round(tempo))


def solo_preprocess(args) -> None:
    audio_path, cfg = args
    track_name = os.path.split(audio_path)[1][:-4]

    audio, original_sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=audio, sr=original_sr)
    song_dur = librosa.get_duration(y=audio, sr=original_sr)

    tempo = get_tempo(
        track_name,
        song_dur,
        cfg.label_dir,
        cfg.output_dir
    )

    process_cqt(
        audio,
        tempo,
        original_sr,
        track_name,
        cfg.split_unit_in_bars,
        cfg.split_hop_bar_len,
        cfg.cqt_hparams,
        cfg.output_dir,
    )

    get_target_file(
        track_name,
        cfg.label_dir,
        cfg.output_dir,
        os.path.abspath(cfg.prediction_dir),
    )

    get_tempo_file(
        track_name,
        tempo,
        cfg.output_dir
    )


@hydra.main(version_base='1.3', config_path='../../conf/data_preprocess', config_name='data_preprocess.yaml')
def main(cfg: DictConfig) -> None:
    audio_filename_list = glob.glob(os.path.join(cfg.audio_dir, "*.wav"))
    for i in range(len(audio_filename_list)):
        solo_preprocess(audio_filename_list[i], cfg)

if __name__ == "__main__":
    main()