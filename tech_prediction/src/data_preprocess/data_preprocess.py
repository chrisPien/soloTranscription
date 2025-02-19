import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
import glob
from omegaconf import DictConfig, OmegaConf
from itertools import repeat
from scipy.spatial.distance import cdist


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
    perdiction_dir,
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
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=hparams["down_sampling_rate"],
        hop_length=hparams["hop_length"],
    )

    for unit_n in range(n_units):
        onset_time_sec_filename = os.path.join(
            output_dir, "onset_time_sec", f"{track_name}_0{unit_n}.npy"
        )
        cqt_filename = os.path.join(
            output_dir, "cqt", f"{track_name}_0{unit_n}.npy"
        )
        # onset_filename = os.path.join(
        #     output_dir, "onset", f"{track_name}_0{unit_n}.npy"
        # )
        frame_level_note_attrib_filename = os.path.join(
            output_dir, "frame_level_note_attrib", f"{track_name}_0{unit_n}.npy"
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

        note_attrib_filename = os.path.join(
            perdiction_dir, f"{track_name}_0{unit_n}.npy"
        )
        tatum_note = np.load(note_attrib_filename)
        onset_tatum = []
        tatum_length = len(tatum_note)
        for i, note in enumerate(tatum_note):
            if note[1] == 1:
                onset_tatum.append(i)
        tatum_frames = [round(tatum * len(onset_env[st:end]) / tatum_length) for tatum in onset_tatum]
        tatum_frames = np.array(tatum_frames)
        cost_matrix = cdist(tatum_frames[:, np.newaxis], np.arange(len(onset_env[st:end]))[:, np.newaxis], metric='euclidean')
        onset_env_tatum = onset_env[tatum_frames]
        onset_cost_matrix = cdist(onset_env_tatum[:, np.newaxis], onset_env[st:end][:, np.newaxis], metric='euclidean')
        alpha = 0.5
        beta = 1 - alpha
        hybrid_cost = alpha * cost_matrix + beta * onset_cost_matrix
        _, wp = librosa.sequence.dtw(C=hybrid_cost)
        #_, wp = librosa.sequence.dtw(C=cost_matrix)

        wp = np.flip(wp, axis=0)
        frame_level_tech = []
        #print(wp)
        # note_onset_time = []
        # current = -1
        for tech, frame in wp:
            frame_level_tech.append(tatum_note[onset_tatum[tech]][2:])
            # if tech != current:
            #     current = tech
        #         note_onset_time.append(frame*256/22050)
        # np.save(onset_time_sec_filename, note_onset_time)
        np.save(cqt_filename, cqt[st:end])
        #np.save(onset_filename, onset_env[st:end])
        #np.save(onset_filename, frame_level_tech)
        np.save(frame_level_note_attrib_filename, frame_level_tech)

        frame_level_pred = os.path.join(
            output_dir, "frame_level_pred", f"{track_name}_0{unit_n}.npy"
        )
        indices = np.linspace(0, len(tatum_note)-1, len(frame_level_tech))
        indices = np.round(indices).astype(int)
        upsampled_pred = [tatum_note[i][2:] for i in indices]
        upsampled_pred = np.array(upsampled_pred)
        np.save(frame_level_pred, upsampled_pred)


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
    # tempo_filename = os.path.join(output_dir, "tempo", f"{track_name}.npy")
    # np.save(tempo_filename, tempo)
    return tempo
    

def get_target_file(
    track_name,
    label_dir,
    output_dir
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

        full_target_filename = os.path.join(
            output_dir, "full_target", f"{track_name}_0{idx}.npy"
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
        target[:, -1] += 1
        np.save(target_filename, target[:, -1])
        frame_level_gt = os.path.join(
            output_dir, "frame_level_gt", f"{track_name}_0{idx}.npy"
        )
        frame_len = len(np.load(os.path.join(output_dir, "cqt", f"{track_name}_0{idx}.npy")))
        indices = np.linspace(0, len(target)-1, frame_len)
        indices = np.round(indices).astype(int)
        upsampled_target = [target[i][2:] for i in indices]
        upsampled_target = np.array(upsampled_target)
        np.save(frame_level_gt, upsampled_target)
        #print(target.shape)

def get_full_target_file(
    track_name,
    full_label_dir,
    output_dir
):
    full_label_filename = os.path.join(full_label_dir, f"{track_name}.tsv")
    with open(full_label_filename, "r") as f:
        lines = [line.split() for line in f.read().split("\n")][:-1]
    f.close()
    full_label = [list(map(int, line)) for line in lines]

    filelist = glob.glob(os.path.join(output_dir, "cqt", f"{track_name}_*.*"))
    
    for file in filelist:
        idx = int((os.path.split(file)[1][:-4]).split("_")[-1])

        full_target_filename = os.path.join(
            output_dir, "full_target", f"{track_name}_0{idx}.npy"
        )

        bar_st = idx
        bar_end = bar_st + 3
        executed = False
        for i in range(len(full_label)):
            if full_label[i][0] == bar_st and executed == False:
                st = i
                executed = True
            if full_label[i][0] == bar_end:
                end = i
        full_target = full_label[st:end+1]
        full_target = np.array(full_target)
        full_target = np.delete(full_target, 0, 1)
        full_target[:, -1] += 1
        #print(full_target)
        np.save(full_target_filename, full_target)
        
        #print(frame_len)

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


# def solo_preprocess(audio_path, cfg) -> None:
def solo_preprocess(args) -> None:
    audio_path, cfg = args
    track_name = os.path.split(audio_path)[1][:-4]
    #print(track_name)

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
        cfg.prediction_dir,
    )

    get_target_file(
        track_name,
        cfg.label_dir,
        cfg.output_dir,
    )

    get_full_target_file(
        track_name,
        cfg.full_label_dir,
        cfg.output_dir,
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
        # if i == 1:
        #     break
        solo_preprocess(audio_filename_list[i], cfg)

if __name__ == "__main__":
    main()