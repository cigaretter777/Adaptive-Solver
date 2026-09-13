import json
import hashlib
import random
from pathlib import Path

import pandas as pd


SEED = 42

INPUT = Path(
    "data/processed/sft_dp_v1_10k.parquet"
)

OUT_DIR = Path(
    "data/processed/sft_dp_v1"
)

MANIFEST = Path(
    "data/manifests/sft_dp_v1_split.json"
)

REPORT = Path(
    "artifacts/reports/sft_dp_v1_quality_report.md"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT)

    print("total:", len(df))


    # deterministic shuffle
    rng = random.Random(SEED)

    indices = list(range(len(df)))
    rng.shuffle(indices)


    split = int(len(df)*0.9)

    train_idx = indices[:split]
    val_idx = indices[split:]


    train = df.iloc[train_idx]
    val = df.iloc[val_idx]


    train_path = OUT_DIR/"train.parquet"
    val_path = OUT_DIR/"validation.parquet"


    train.to_parquet(train_path,index=False)
    val.to_parquet(val_path,index=False)


    manifest = {

        "dataset_version":
            "sft-dp-v1",

        "source":
            str(INPUT),

        "seed":
            SEED,

        "total":
            len(df),

        "train":
            {
                "count":len(train),
                "file":str(train_path),
                "sha256":sha256(train_path)
            },

        "validation":
            {
                "count":len(val),
                "file":str(val_path),
                "sha256":sha256(val_path)
            }
    }


    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2
        )
    )


    report=f"""
# SFT Dataset v1 Quality Report


## Dataset

source:

{INPUT}


total:

{len(df)}


## Split


train:

{len(train)}


validation:

{len(val)}


seed:

{SEED}


## Hash


train:

{sha256(train_path)}


validation:

{sha256(val_path)}

"""


    REPORT.write_text(report)


    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
