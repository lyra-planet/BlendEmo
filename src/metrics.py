"""Filename parsing and evaluation metrics for BlendEmo."""

import numpy as np


def parse_filename(filename):
    metadata = filename.split("_")
    video_id = metadata[0]
    item = {"filename": filename, "video_id": video_id}

    temp_mix = metadata[1]
    if temp_mix == "mix":
        item.update(
            {
                "mix": 1,
                "emotion_1": metadata[2],
                "emotion_2": metadata[3],
                "emotion_1_salience": metadata[4],
                "emotion_2_salience": metadata[5],
                "version": metadata[6][3],
            }
        )
    else:
        item.update(
            {
                "mix": 0,
                "emotion_1": metadata[1],
                "version": metadata[3][3],
            }
        )
        sit_int = metadata[2]
        if sit_int[0:3] == "int":
            item["intensity_level"] = sit_int[3]
        elif sit_int[0:3] == "sit":
            item["situation"] = sit_int[3]

    return item


def metadata_to_label(metadata):
    filename = metadata.get("filename")
    emotion_1 = metadata.get("emotion_1")
    emotion_2 = metadata.get("emotion_2")

    if emotion_2:
        return {
            filename: [
                {"emotion": emotion_1, "salience": metadata["emotion_1_salience"]},
                {"emotion": emotion_2, "salience": metadata["emotion_2_salience"]},
            ]
        }

    return {filename: [{"emotion": emotion_1, "salience": 1.0}]}


def acc_presence_single(label, pred):
    label_emotions = {l["emotion"] for l in label}
    pred_emotions = {p["emotion"] for p in pred}
    return label_emotions == pred_emotions


def acc_salience_single(label, pred):
    if len(label) != 2 or len(pred) != 2:
        raise ValueError("Both label and prediction must contain exactly two emotions.")
    label_dict = {l["emotion"]: round(float(l["salience"])) for l in label}
    pred_dict = {p["emotion"]: round(float(p["salience"])) for p in pred}
    return label_dict == pred_dict


def acc_presence_total(preds):
    res = []
    for filename, predictions in preds.items():
        label = metadata_to_label(parse_filename(filename))[filename]
        res.append(acc_presence_single(label, predictions))
    return np.mean(res).item()


def acc_salience_total(preds):
    res = []
    for filename, predictions in preds.items():
        label = metadata_to_label(parse_filename(filename))[filename]
        if len(label) != 2:
            continue
        if len(predictions) == 2:
            res.append(acc_salience_single(label, predictions))
        else:
            res.append(False)
    if not res:
        return 0.0
    return np.mean(res).item()
