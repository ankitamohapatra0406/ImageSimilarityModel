# Image Similarity Detection System

This project implements a Siamese Neural Network for comparing a reference image with a generated image and calculating similarity scores.

## Features

- Image preprocessing
- Feature extraction
- Similarity calculation
- Model training
- API integration

---

## Contest Quick Start

The `contest_scorer.py` script is a self-contained tool for scoring and ranking participants in an AI image generation contest.

### Setup

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Folder Structure

Organize your contest files like this:

```
contest/
├── references/          # Reference images
│   ├── ref_01.png
│   ├── ref_02.png
│   └── ...
├── submissions/         # One folder per participant
│   ├── Alice/
│   │   ├── sub_01.png   # Alice's attempt at ref_01
│   │   ├── sub_02.png
│   │   └── ...
│   ├── Bob/
│   │   ├── sub_01.png
│   │   └── sub_02.png
│   └── ...
└── results/             # Auto-created by the script
    ├── leaderboard.csv
    └── detailed_scores.csv
```

**Naming convention:**
- Reference images: `ref_01.png`, `ref_02.png`, ... (any image extension)
- Submission images: `sub_01.png`, `sub_02.png`, ... (number must match the reference)

### Usage

```bash
# Score the entire contest and generate the leaderboard
python contest_scorer.py --refs contest/references --subs contest/submissions --out contest/results

# Also output JSON (for API integration)
python contest_scorer.py --refs contest/references --subs contest/submissions --out contest/results --json

# Quick comparison of two images
python contest_scorer.py --pair image_a.png image_b.png

# Use a different model checkpoint
python contest_scorer.py --refs refs/ --subs subs/ --model path/to/model.pth
```

### Output

**`leaderboard.csv`** — Ranked results with average score per participant:

| Rank | Participant | Avg Score | Images Scored | Images Expected |
|------|-------------|-----------|---------------|-----------------|
| 1    | Alice       | 87.34     | 50            | 50              |
| 2    | Bob         | 82.10     | 50            | 50              |

**`detailed_scores.csv`** — Per-image breakdown:

| Participant | Image ID | Ref File   | Sub File   | Score |
|-------------|----------|------------|------------|-------|
| Alice       | 1        | ref_01.png | sub_01.png | 91.23 |
| Alice       | 2        | ref_02.png | sub_02.png | 83.45 |

---

## Training (Advanced)

The model training pipeline lives in `src/`. You only need this if you want to retrain the model.

```bash
cd src
python train.py
python evaluate.py
```
