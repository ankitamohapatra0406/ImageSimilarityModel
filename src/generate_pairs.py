import os
import random
import pandas as pd

random.seed(42)


def extract_number(filename):
    return int(filename.split("_")[1].split(" ")[0].split(".")[0])


def generate_pairs(original_dir, generated_dir, output_csv):

    originals = sorted(os.listdir(original_dir))
    generated = sorted(os.listdir(generated_dir))

    pairs = []

    for org in originals:

        org_num = extract_number(org)

        # Positive Pair
        positive = None

        for gen in generated:
            if extract_number(gen) == org_num:
                positive = gen
                break

        if positive is not None:
            pairs.append([
                os.path.join(original_dir, org),
                os.path.join(generated_dir, positive),
                1
            ])

      
        # Hard Negatives
        hard_negatives = []

        nearby_ids = {
            org_num - 2,
            org_num - 1,
            org_num + 1,
            org_num + 2
        }

        for gen in generated:

            gen_num = extract_number(gen)

            if gen_num in nearby_ids:
                hard_negatives.append(gen)

        # Maximum 4 hard negatives
        hard_negatives = hard_negatives[:4]

    
        # Random Negatives
        remaining = []

        for gen in generated:

            gen_num = extract_number(gen)

            if (
                gen_num != org_num
                and gen not in hard_negatives
            ):
                remaining.append(gen)

        random.shuffle(remaining)

        random_negatives = remaining[:4]

        
        # Save Hard Negatives
        for gen in hard_negatives:

            pairs.append([
                os.path.join(original_dir, org),
                os.path.join(generated_dir, gen),
                0
            ])

        # Save Random Negatives
        for gen in random_negatives:

            pairs.append([
                os.path.join(original_dir, org),
                os.path.join(generated_dir, gen),
                0
            ])

    random.shuffle(pairs)

    df = pd.DataFrame(
        pairs,
        columns=["img1", "img2", "label"]
    )

    df.to_csv(output_csv, index=False)

    print(f"\n{output_csv} created")
    print(df.head())
    print(f"Total pairs: {len(df)}")


generate_pairs(
    "dataset/train_original",
    "dataset/train_generated",
    "dataset/train_pairs.csv"
)

generate_pairs(
    "dataset/test_original",
    "dataset/test_generated",
    "dataset/test_pairs.csv"
)