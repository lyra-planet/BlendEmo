"""Label definitions for BlendEmo."""

LABEL_TO_INDEX = {
    "ang": 0,
    "disg": 1,
    "fea": 2,
    "hap": 3,
    "sad": 4,
    "neu": 5,
}

NEUTRAL_INDEX = LABEL_TO_INDEX["neu"]
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
