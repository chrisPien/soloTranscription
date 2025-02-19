import glob
import os
import numpy as np
from sklearn.metrics import f1_score
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

gt_path = "D:/solo_tech/data/preprocessed/frame_level_gt"
#pred_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-09-17_21-27-08/frame_level_full_tech_prediction"
pred_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-09-19_09-21-15/frame_level_full_tech_prediction"
# considered in loss function
pred_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-09-25_13-23-48/frame_level_full_tech_prediction"
prediction_path = "D:/solo_tech/logs/train_hybrid_ctc/runs/2024-10-13_02-19-33/frame_level_full_tech_prediction"

attrib_pred_path = "D:/solo_tech/data/preprocessed/frame_level_pred"

file_name = glob.glob(os.path.join(gt_path, "*npy"))
#file_name = glob.glob(os.path.join(prediction_path, "*tsv"))
tp_list = [0, 0, 0]
fn_list = [0, 0, 0]
fp_list = [0, 0, 0]

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
inverted_group_dict = {value: key for key, values in group_dict.items() for value in values}
inverted_tech_dict = {value: key for key, values in tech_dict.items() for value in values}

group_tp_list = [0] * 4
group_fn_list = [0] * 4
group_fp_list = [0] * 4

tech_tp_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
tech_fn_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
tech_fp_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

final_tech_tp_list = [0] * 24
final_tech_fn_list = [0] * 24
final_tech_fp_list = [0] * 24

# files = []
# for file in file_name:
#     idx = (os.path.split(file)[1][:-31])
#     files.append(os.path.join("D:/solo_tech/data/preprocessed/full_target/" + idx + ".npy"))
# print(files)
file_name = glob.glob(os.path.join(prediction_path, "*tsv"))
files = []
for file in file_name:
    idx = (os.path.split(file)[1][:-29])
    files.append(os.path.join("D:/solo_tech/data/preprocessed/frame_level_gt/" + idx + ".npy"))

for file in files:
    #print(file)
    gt = np.load(file)
    idx = (os.path.split(file)[1][:-4])
    with open(os.path.join(prediction_path, idx + "_frame_level_note_attribs.tsv"), 'r') as f:
        lines = [line.strip()[1:-1].split(', ') for line in f.read().split("\n")][:-1]
    f.close()
    pred = [list(map(int, line)) for line in lines]
    pred = np.array(pred)
    attrib_pred = np.load(os.path.join(attrib_pred_path, idx + ".npy"))
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

    tp, fn, fp = calculate_tp_fn_fp(gt, attrib_pred, [1, 2], onset_matching)
    tp_list[1] += tp
    fp_list[1] += fp
    fn_list[1] += fn

    tp_list[2] += len(tech_matching)
    fp_list[2] += len(pred) - len(tech_matching)
    fn_list[2] += len(gt) - len(tech_matching)

    for gt_idx, pred_idx in tech_matching:
        tech_tp_list[inverted_tech_dict[gt[gt_idx, -1]] - 1] += 1  # True Positive: matching keys and values
        final_tech_tp_list[gt[gt_idx, -1] - 1] += 1
        # group_tp_list[inverted_group_dict[inverted_tech_dict[gt[gt_idx, -1]]] - 1] += 1


    all_gt_indices = set(range(len(gt)))
    matched_gt_indices = {gt_idx for gt_idx, _ in tech_matching}
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    for gt_idx in unmatched_gt_indices:
        tech_fn_list[inverted_tech_dict[gt[gt_idx, -1]] - 1] += 1
        final_tech_fn_list[gt[gt_idx, -1] - 1] += 1
        # group_fn_list[inverted_group_dict[inverted_tech_dict[gt[gt_idx, -1]]] - 1] += 1

    all_pred_indices = set(range(len(pred)))
    matched_pred_indices = {pred_idx for _, pred_idx in tech_matching}
    unmatched_pred_indices = all_pred_indices - matched_pred_indices
    for pred_idx in unmatched_pred_indices:
        tech_fp_list[inverted_tech_dict[pred[pred_idx, -1]] - 1] += 1
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
f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(tp_list, fp_list, fn_list)]
# precision = tp / (tp + fp + 1e-7)
# recall = tp / (tp + fn + 1e-7)
# f1 = 2 * precision * recall / (precision + recall + 1e-7)
print(f1s)

group_f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(group_tp_list, group_fp_list, group_fn_list)]
print(group_f1s)

tech_f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(tech_tp_list, tech_fp_list, tech_fn_list)]
print(tech_f1s)
print(tech_tp_list, tech_fp_list, tech_fn_list)

final_tech_f1s = [calculate_f1(tp, fp, fn) for tp, fp, fn in zip(final_tech_tp_list, final_tech_fp_list, final_tech_fn_list)]
print(final_tech_f1s)