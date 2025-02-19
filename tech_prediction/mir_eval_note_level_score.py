import numpy as np
import glob
import os
import mir_eval

def calculate_tp_fn_fp(gt, pred, indices, matching):
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

def calculate_f1(tp, fn, fp):
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    if tp == 0 and fn == 0 and fp == 0:
        return -1
    return f1

target_path = "D:/solo_tech/data/preprocessed/full_target"

prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_02-19-33/full_tech_prediction"
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_02-19-33/tech_prediction"
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_02-19-33/group_prediction"

# # group category tech
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_20-05-35/full_tech_prediction"
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_20-05-35/tech_prediction"
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_20-05-35/group_prediction"
# # group tech
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_20-43-13/full_tech_prediction"
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_20-43-13/group_prediction"
# # category tech
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_21-22-44/full_tech_prediction"
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_21-22-44/tech_prediction"
# # direct tech
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_21-59-57/full_tech_prediction"
# # note attributes
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_22-14-15/full_tech_prediction"
# # upsampling instead dtw
# prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_22-32-04/full_tech_prediction"
# # no bpm
# prediction_path = "D:/solo_tech/ablation_study_no_bpm"
# # note-level label
# prediction_path = "D:/solo_tech/ablation_study_note_level"
# # psf seperate
# prediction_path = "D:/solo_tech/ablation_study_p_s_f_seperate"
# # tech in conformer
# prediction_path = "D:/solo_tech/ablation_study_tech_in_trans"


file_name = glob.glob(os.path.join(prediction_path, "*tsv"))
files = []
for file in file_name:
    '''-------------------------------------------------------------------'''
    idx = (os.path.split(file)[1][:-31])
    # idx = (os.path.split(file)[1][:-25])
    # idx = (os.path.split(file)[1][:-26])
    files.append(os.path.join("D:/solo_tech/data/preprocessed/full_target/" + idx + ".npy"))


indices_map = {
    0: 'pitch, onset',
    1: 'pitch, onset, dur',
    2: 'pitch, onset, dur, tech',
    3: 'string, fret, onset',
    4: 'string, fret, onset, dur',
    5: 'string, fret, onset, dur, tech',
    6: 'onset, tech',
}
tp_list = [0] * len(indices_map)
fn_list = [0] * len(indices_map)
fp_list = [0] * len(indices_map)

group_dict = {
    0: [0],
    1: [1],
    2: [2, 3, 4],
    3: [5, 6, 7],
    4: [8, 9]
}

tech_dict = {
    0: [0],
    1: [1, 19], # normal
    2: [2, 3, 4], # slide
    3: [5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 24], # bend
    4: [13], # trill
    5: [17], # hammer
    6: [15], # pull
    7: [18], # tap
    8: [16], # harmonic
    9: [14, 23] # mute
}
inverted_tech_dict = {value: key for key, values in tech_dict.items() for value in values}
inverted_group_dict = {value: key for key, values in group_dict.items() for value in values}

group_tp_list = [0] * 4
group_fn_list = [0] * 4
group_fp_list = [0] * 4

tech_tp_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
tech_fn_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
tech_fp_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

final_tech_tp_list = [0] * 24
final_tech_fn_list = [0] * 24
final_tech_fp_list = [0] * 24

for file in files:
    target = np.load(file)
    accumulated_value = -48
    for i in range(target.shape[0]):
        if target[i, 0] == 0:
            accumulated_value += 48
        target[i, 0] += accumulated_value

    idx = (os.path.split(file)[1][:-4])
    '''-----------------------------------------------------------------------------------------------'''
    with open(os.path.join(prediction_path, idx + "_final_tech_full_prediction.tsv"), 'r') as f:
    # with open(os.path.join(prediction_path, idx + "_tech_full_prediction.tsv"), 'r') as f:
    # with open(os.path.join(prediction_path, idx + "_group_full_prediction.tsv"), 'r') as f:
        lines = [line.strip()[1:-1].split(', ') for line in f.read().split("\n")][:-1]
    f.close()
    pred = [list(map(int, line)) for line in lines]
    pred = np.array(pred)
    accumulated_value = -48
    for i in range(pred.shape[0]):
        if pred[i, 0] == 0:
            accumulated_value += 48
        pred[i, 0] += accumulated_value
    #print(pred)

    tempo = np.load(os.path.join("D:/solo_tech/data/preprocessed/tempo", idx + ".npy"))
    #print(tempo)

    #new_column = (target[:, 0] + 1) / 100
    #new_column = (target[:, 0] + target[:, 1]) / 100
    new_column = (target[:, 0] + target[:, 1]) * ((4 * 60) / (tempo * 48))
    #reference_onsets = np.column_stack((target[:, 0] / 100, new_column))
    reference_onsets = np.column_stack((target[:, 0] * ((4 * 60) / (tempo * 48)), new_column))
    adjusted_midi = [0 if midi == 100 else 101 if midi == 101 else midi - 12 for midi in target[:, 2]]
    reference_pitches = np.array([mir_eval.util.midi_to_hz(midi) for midi in adjusted_midi])
    reference_tech = target[:, -1]
    '''-------------------------------------------------------------------------------'''
    # reference_tech = [inverted_tech_dict[tech] for tech in reference_tech]
    # target[:, -1] = [inverted_tech_dict[tech] for tech in target[:, -1]]
    # reference_tech = [inverted_group_dict[inverted_tech_dict[tech]] for tech in reference_tech]
    # target[:, -1] = [inverted_group_dict[inverted_tech_dict[tech]] for tech in target[:, -1]]
    #print(reference_onsets)

    #new_column = (pred[:, 0] + 1) / 100
    #new_column = (pred[:, 0] + pred[:, 1]) / 100
    new_column = (pred[:, 0] + pred[:, 1]) * ((4 * 60) / (tempo * 48))
    #estimated_onsets = np.column_stack((pred[:, 0] / 100, new_column))
    estimated_onsets = np.column_stack((pred[:, 0] * ((4 * 60) / (tempo * 48)), new_column))
    adjusted_midi = [0 if midi == 100 else 101 if midi == 101 else midi - 12 for midi in pred[:, 2]]
    estimated_pitches = np.array([mir_eval.util.midi_to_hz(midi) for midi in adjusted_midi])
    estimated_tech = pred[:, -1]
    #print(estimated_onsets)


    onset_matching = mir_eval.transcription.match_notes(
        reference_onsets,
        reference_pitches,
        estimated_onsets,
        estimated_pitches,
        #onset_tolerance=0.03,
        #pitch_tolerance=0,
        offset_ratio=None
    )

    onset_offset_matching = mir_eval.transcription.match_notes(
        reference_onsets,
        reference_pitches,
        estimated_onsets,
        estimated_pitches,
        #onset_tolerance=0.03,
        offset_min_tolerance=0.03,
        #pitch_tolerance=0,
    )

    tech_matching = mir_eval.transcription.match_notes(
        reference_onsets,
        reference_tech,
        estimated_onsets,
        estimated_tech,
        #onset_tolerance=0.03,
        pitch_tolerance=0,
        offset_ratio=None
    )

    # pitch, onset
    tp_list[0] += len(onset_matching)
    fp_list[0] += len(pred) - len(onset_matching)
    fn_list[0] += len(target) - len(onset_matching)

    #pitch, onset, dur
    tp_list[1] += len(onset_offset_matching)
    fp_list[1] += len(pred) - len(onset_offset_matching)
    fn_list[1] += len(target) - len(onset_offset_matching)

    #pitch, onset, dur, tech
    tp, fn, fp = calculate_tp_fn_fp(target, pred, [5], onset_offset_matching)
    tp_list[2] += tp
    fp_list[2] += fp
    fn_list[2] += fn

    #string, fret, onset
    tp, fn, fp = calculate_tp_fn_fp(target, pred, [3, 4], onset_matching)
    tp_list[3] += tp
    fp_list[3] += fp
    fn_list[3] += fn

    #string, fret, onset, dur
    tp, fn, fp = calculate_tp_fn_fp(target, pred, [3, 4], onset_offset_matching)
    tp_list[4] += tp
    fp_list[4] += fp
    fn_list[4] += fn

    #string, fret, onset, dur
    tp, fn, fp = calculate_tp_fn_fp(target, pred, [3, 4, 5], onset_offset_matching)
    tp_list[5] += tp
    fp_list[5] += fp
    fn_list[5] += fn

    #onset, tech
    # tp, fn, fp = calculate_tp_fn_fp(target, pred, [5], onset_matching)
    # tp_list[6] += tp
    # fp_list[6] += fp
    # fn_list[6] += fn
    tp_list[6] += len(tech_matching)
    fp_list[6] += len(pred) - len(tech_matching)
    fn_list[6] += len(target) - len(tech_matching)

    #all techs
    for gt_idx, pred_idx in tech_matching:
        # tech_tp_list[inverted_tech_dict[target[gt_idx, -1]] - 1] += 1  # True Positive: matching keys and values
        # final_tech_tp_list[target[gt_idx, -1] - 1] += 1
        # group_tp_list[inverted_group_dict[inverted_tech_dict[target[gt_idx, -1]]] - 1] += 1
        tech_tp_list[inverted_tech_dict[reference_tech[gt_idx]] - 1] += 1  # True Positive: matching keys and values
        final_tech_tp_list[reference_tech[gt_idx] - 1] += 1
        group_tp_list[inverted_group_dict[inverted_tech_dict[reference_tech[gt_idx]]] - 1] += 1

    all_gt_indices = set(range(len(target)))
    matched_gt_indices = {gt_idx for gt_idx, _ in tech_matching}
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    for gt_idx in unmatched_gt_indices:
        # tech_fn_list[inverted_tech_dict[target[gt_idx, -1]] - 1] += 1
        # final_tech_fn_list[target[gt_idx, -1] - 1] += 1
        # group_fn_list[inverted_group_dict[inverted_tech_dict[target[gt_idx, -1]]] - 1] += 1
        tech_fn_list[inverted_tech_dict[reference_tech[gt_idx]] - 1] += 1
        final_tech_fn_list[reference_tech[gt_idx] - 1] += 1
        group_fn_list[inverted_group_dict[inverted_tech_dict[reference_tech[gt_idx]]] - 1] += 1

    all_pred_indices = set(range(len(pred)))
    matched_pred_indices = {pred_idx for _, pred_idx in tech_matching}
    unmatched_pred_indices = all_pred_indices - matched_pred_indices
    for pred_idx in unmatched_pred_indices:
        # tech_fp_list[inverted_tech_dict[pred[pred_idx, -1]] - 1] += 1
        # final_tech_fp_list[pred[pred_idx, -1] - 1] += 1
        # group_fp_list[inverted_group_dict[inverted_tech_dict[pred[pred_idx, -1]]] - 1] += 1
        tech_fp_list[inverted_tech_dict[estimated_tech[pred_idx]] - 1] += 1
        final_tech_fp_list[estimated_tech[pred_idx] - 1] += 1
        group_fp_list[inverted_group_dict[inverted_tech_dict[estimated_tech[pred_idx]]] - 1] += 1
    #break

f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(tp_list, fp_list, fn_list)]
#print(f1s)
for idx, metrics in indices_map.items():
    print(metrics, ":", f1s[idx])
# print(tp_list[1], fp_list[1], fn_list[1], tp_list[1] * 2 + fp_list[1] + fn_list[1])
# print(tp_list[4], fp_list[4], fn_list[4], tp_list[4] * 2 + fp_list[4] + fn_list[4])
# print(tp_list[2], fp_list[2], fn_list[2], tp_list[2] * 2 + fp_list[2] + fn_list[2])

group_f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(group_tp_list, group_fp_list, group_fn_list)]
print(group_f1s)

tech_f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(tech_tp_list, tech_fp_list, tech_fn_list)]
print(tech_f1s)
print(tech_tp_list, tech_fp_list, tech_fn_list)

final_tech_f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(final_tech_tp_list, final_tech_fp_list, final_tech_fn_list)]
print(final_tech_f1s)
# print(final_tech_tp_list)
# print(final_tech_fp_list)
# print(final_tech_fn_list)
print([2*a+b+c for a, b, c in zip(final_tech_tp_list, final_tech_fp_list, final_tech_fn_list)])
print([a+b+c for a, b, c in zip(final_tech_tp_list, final_tech_fp_list, final_tech_fn_list)])
# print(sum(tech_tp_list), sum(tech_fp_list), sum(tech_fn_list), sum(tech_tp_list) * 2 + sum(tech_fp_list) + sum(tech_fn_list))