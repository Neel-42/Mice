# Mouse ECoG seizure / SWD detection

This repository contains code to work with the two MATLAB v7.3 recordings in the project folder. Each file has a `rec` group with:

| Field       | Meaning                                      |
|------------|-----------------------------------------------|
| `ecog`     | Continuous ECoG (float vector)               |
| `SWDlabel` | Per-sample label codes (0 = baseline; >0 spike-wave–related activity in this dataset) |
| `fs`       | Sample rate (200 Hz in your files)           |
| `mID`      | Mouse identifier (30)                        |

The machine-learning piece **detects periods that match your lab’s `SWDlabel` coding** using sliding windows of spectral and time-domain features plus a random forest classifier.

## Important limits on “genes causing seizure types”

Your folder currently has **two `.mat` recordings**, not separate gene-expression or genotype documents. DNA / RNA information is not inside those files.

To connect **genes → seizure phenotypes**, this project includes:

1. `data/gene_seizure_associations.csv` — a small **literature-based starter table** (human/mouse epilepsy genes and rough phenotypes). Replace or extend it with your own references.
2. `data/mouse_metadata.example.json` — copy to e.g. `data/mouse_metadata.json`, set real genotypes per `mID`, then use `src/report.py` to print matching rows from the CSV.

Inferring causality from genotype to a specific SWD subtype **requires your experimental design** (knockout strain, breeding, histology, etc.). The code only **joins metadata you supply** to the curated table; it does not discover new gene–seizure links from ECoG alone.

Update `label_code_to_phenotype` in `src/gene_phenotype.py` once you have the official legend for label integers `1,3,5,6` from your lab.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Summarize labels (and optional genotype hints):

```bash
python3 -m src.report --mat "m30_Rec1_ALL (1).mat" \
  --metadata-json data/mouse_metadata.example.json
```

Train (full recordings — can take a while; uses temporal split so last part of the stacked windows is held out):

```bash
python3 -m src.train_model --mat "m30_Rec1_ALL (1).mat" "m30_Rec2_ALL (1).mat" \
  --out models/seizure_rf.joblib
```

**Binary** (any SWD vs baseline; uses max label inside each window):

```bash
python3 -m src.train_model --mat "m30_Rec1_ALL (1).mat" "m30_Rec2_ALL (1).mat" \
  --binary --out models/seizure_binary_rf.joblib
```

Faster dry run on the first *N* seconds of each file:

```bash
python3 -m src.train_model --mat "m30_Rec1_ALL (1).mat" "m30_Rec2_ALL (1).mat" \
  --max-seconds 7200 --stride-s 2 --out models/seizure_rf_dev.joblib
```

Apply the saved model:

```bash
python3 -m src.predict --mat "m30_Rec1_ALL (1).mat" --model models/seizure_rf.joblib \
  --out-csv outputs/predictions_rec1.csv
```

Binary outputs include `swdlabel_max_in_window` for reference.

## Outputs

- `models/seizure_rf.joblib` — trained model bundle (also writes `.meta.json`).
- `outputs/predictions*.csv` — per-window start time, true label aggregate, predicted class, class probabilities.

## File naming

If you rename the `.mat` files, pass the new paths to `--mat` / `--mat` flags; the loader only requires the internal `rec` layout described above.
