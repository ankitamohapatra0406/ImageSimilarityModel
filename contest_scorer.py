"""
Contest Image Similarity Scorer
================================
Scores participant submissions against reference images using a trained
Siamese Network and produces a ranked leaderboard.

Naming convention:
  - Reference images : ref_01.png, ref_02.png, ...
  - Submission images: sub_01.png, sub_02.png, ...

Folder layout:
  contest/
  ├── references/        # ref_01.png, ref_02.png, ...
  ├── submissions/       # One folder per participant
  │   ├── Alice/
  │   │   ├── sub_01.png
  │   │   └── sub_02.png
  │   └── Bob/
  │       ├── sub_01.png
  │       └── sub_02.png
  └── results/           # Auto-created
      ├── leaderboard.csv
      └── detailed_scores.csv

Usage:
  # Full contest scoring
  python contest_scorer.py --refs contest/references --subs contest/submissions --out contest/results

  # Quick pair check
  python contest_scorer.py --pair image_a.png image_b.png

  # JSON output (for API integration)
  python contest_scorer.py --refs ... --subs ... --out ... --json
"""

import os
import re
import sys
import csv
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import the Siamese model from the existing src/ package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model import SiameseNetwork  # noqa: E402


# ============================= Helpers =====================================

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Matches filenames like ref_01.png, ref_2.jpg, ref_100.webp
REF_PATTERN = re.compile(r"^ref_(\d+)", re.IGNORECASE)
# Matches filenames like sub_01.png, sub_2.jpg, sub_100.webp
SUB_PATTERN = re.compile(r"^sub_(\d+)", re.IGNORECASE)


def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, device):
    """Load the trained Siamese Network."""
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at '{model_path}'")
        print("       Run train.py first, or pass --model with the correct path.")
        sys.exit(1)

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def make_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def load_image(path, transform, device):
    """Load, preprocess, and move a single image to the target device."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


@torch.no_grad()
def get_embedding(model, tensor):
    """Extract a feature embedding from an image tensor."""
    return model.encoder(tensor).cpu().numpy()


def compute_score(emb1, emb2):
    """Cosine similarity mapped to a 0-100 percentage score."""
    sim = cosine_similarity(emb1, emb2)[0][0]
    return round(float((sim + 1) / 2 * 100), 2)


# ========================= File Discovery =================================

def discover_references(ref_dir):
    """
    Scan the reference directory and return a dict mapping image ID → filepath.
    e.g. { 1: '/path/to/ref_01.png', 2: '/path/to/ref_02.png', ... }
    """
    ref_map = {}
    for fname in os.listdir(ref_dir):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        match = REF_PATTERN.match(fname)
        if match:
            img_id = int(match.group(1))
            ref_map[img_id] = os.path.join(ref_dir, fname)
    return ref_map


def discover_submissions(subs_dir):
    """
    Scan the submissions directory.
    Returns a dict: { participant_name: { image_id: filepath, ... }, ... }
    """
    participants = {}
    for entry in sorted(os.listdir(subs_dir)):
        participant_dir = os.path.join(subs_dir, entry)
        if not os.path.isdir(participant_dir):
            continue

        images = {}
        for fname in os.listdir(participant_dir):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            match = SUB_PATTERN.match(fname)
            if match:
                img_id = int(match.group(1))
                images[img_id] = os.path.join(participant_dir, fname)

        if images:
            participants[entry] = images

    return participants


# ============================== Modes =====================================

def run_pair(img1_path, img2_path, model_path):
    """Quick comparison of two images — prints the similarity score."""
    device = get_device()
    model = load_model(model_path, device)
    transform = make_transform()

    for p in (img1_path, img2_path):
        if not os.path.exists(p):
            print(f"ERROR: Image not found — {p}")
            sys.exit(1)

    emb1 = get_embedding(model, load_image(img1_path, transform, device))
    emb2 = get_embedding(model, load_image(img2_path, transform, device))

    score = compute_score(emb1, emb2)
    print(f"\n  Image 1 : {img1_path}")
    print(f"  Image 2 : {img2_path}")
    print(f"  Score   : {score:.2f}%\n")
    return score


def run_contest(ref_dir, subs_dir, out_dir, model_path, output_json=False):
    """
    Full contest scoring pipeline.
    Compares every participant's submissions against the references,
    computes per-image and average scores, and writes results.
    """
    # --- Validate directories ---
    for d, label in [(ref_dir, "References"), (subs_dir, "Submissions")]:
        if not os.path.isdir(d):
            print(f"ERROR: {label} directory not found — {d}")
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}\n")

    model = load_model(model_path, device)
    transform = make_transform()

    # --- Discover files ---
    ref_map = discover_references(ref_dir)
    if not ref_map:
        print(f"ERROR: No reference images found in '{ref_dir}'")
        print("       Expected filenames like ref_01.png, ref_02.png, ...")
        sys.exit(1)

    participants = discover_submissions(subs_dir)
    if not participants:
        print(f"ERROR: No participant submissions found in '{subs_dir}'")
        print("       Expected subfolders with files like sub_01.png, sub_02.png, ...")
        sys.exit(1)

    num_refs = len(ref_map)
    num_participants = len(participants)
    print(f"Found {num_refs} reference images")
    print(f"Found {num_participants} participants\n")

    # --- Pre-compute reference embeddings (compute once, reuse) ---
    print("Encoding reference images...")
    ref_embeddings = {}
    for img_id in tqdm(sorted(ref_map.keys()), desc="References"):
        tensor = load_image(ref_map[img_id], transform, device)
        ref_embeddings[img_id] = get_embedding(model, tensor)

    # --- Score each participant ---
    detailed_rows = []  # Per-image scores
    leaderboard = []    # Per-participant average

    print("\nScoring submissions...")
    for name in tqdm(sorted(participants.keys()), desc="Participants"):
        sub_images = participants[name]
        scores = []
        matched_ids = sorted(set(sub_images.keys()) & set(ref_embeddings.keys()))

        if not matched_ids:
            print(f"  WARNING: '{name}' has no matching image IDs — skipped")
            continue

        for img_id in matched_ids:
            try:
                sub_tensor = load_image(sub_images[img_id], transform, device)
                sub_emb = get_embedding(model, sub_tensor)
                score = compute_score(ref_embeddings[img_id], sub_emb)
                scores.append(score)
                detailed_rows.append({
                    "participant": name,
                    "image_id": img_id,
                    "ref_file": os.path.basename(ref_map[img_id]),
                    "sub_file": os.path.basename(sub_images[img_id]),
                    "score": score,
                })
            except Exception as e:
                print(f"  WARNING: Failed to score {name}/sub_{img_id}: {e}")

        if scores:
            avg = round(sum(scores) / len(scores), 2)
            leaderboard.append({
                "participant": name,
                "avg_score": avg,
                "images_scored": len(scores),
                "images_expected": num_refs,
            })

    # --- Sort leaderboard (highest score first) ---
    leaderboard.sort(key=lambda x: x["avg_score"], reverse=True)

    # --- Assign ranks ---
    for rank, entry in enumerate(leaderboard, 1):
        entry["rank"] = rank

    # --- Print leaderboard to console ---
    print("\n" + "=" * 70)
    print("  CONTEST LEADERBOARD")
    print("=" * 70)
    print(f"  {'Rank':<6}{'Participant':<30}{'Avg Score':<14}{'Images':<10}")
    print("-" * 70)
    for entry in leaderboard:
        imgs = f"{entry['images_scored']}/{entry['images_expected']}"
        print(f"  {entry['rank']:<6}{entry['participant']:<30}{entry['avg_score']:<14.2f}{imgs:<10}")
    print("=" * 70)

    # --- Write CSV files ---
    leaderboard_csv = os.path.join(out_dir, "leaderboard.csv")
    with open(leaderboard_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "participant", "avg_score", "images_scored", "images_expected"])
        writer.writeheader()
        writer.writerows(leaderboard)
    print(f"\n  Leaderboard saved to: {leaderboard_csv}")

    detailed_csv = os.path.join(out_dir, "detailed_scores.csv")
    with open(detailed_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant", "image_id", "ref_file", "sub_file", "score"])
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"  Detailed scores saved to: {detailed_csv}")

    # --- Optional JSON output ---
    if output_json:
        json_path = os.path.join(out_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump({
                "leaderboard": leaderboard,
                "detailed_scores": detailed_rows,
            }, f, indent=2)
        print(f"  JSON results saved to: {json_path}")

    print()
    return leaderboard


# ============================= CLI ========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Contest Image Similarity Scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score an entire contest
  python contest_scorer.py --refs contest/references --subs contest/submissions

  # Quick pair comparison
  python contest_scorer.py --pair imageA.png imageB.png

  # With JSON output
  python contest_scorer.py --refs refs/ --subs subs/ --out results/ --json
        """,
    )

    # Contest mode
    parser.add_argument("--refs", "-r", default=None,
                        help="Directory containing reference images (ref_01.png, ...)")
    parser.add_argument("--subs", "-s", default=None,
                        help="Directory containing participant submission folders")
    parser.add_argument("--out", "-o", default="contest_results",
                        help="Output directory for CSV/JSON results (default: contest_results)")
    parser.add_argument("--json", action="store_true",
                        help="Also output results as JSON")

    # Quick pair mode
    parser.add_argument("--pair", nargs=2, metavar=("IMG1", "IMG2"),
                        help="Compare two images directly and print the score")

    # Model path
    parser.add_argument("--model", "-m", default="saved_models/siamese_model.pth",
                        help="Path to trained model weights (default: saved_models/siamese_model.pth)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.pair:
        run_pair(args.pair[0], args.pair[1], args.model)

    elif args.refs and args.subs:
        run_contest(args.refs, args.subs, args.out, args.model, output_json=args.json)

    else:
        print("ERROR: Specify either --pair for quick comparison or --refs + --subs for contest scoring.")
        print("       Run with --help for usage details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
