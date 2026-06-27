import os
import random
import pandas as pd

random.seed(42)


def generate_triplets(original_dir, generated_dir, output_csv):

    originals = sorted(os.listdir(original_dir))
    generated = sorted(os.listdir(generated_dir))

    triplets = []

    for org in originals:

        org_num = int(org.split("_")[1].split(".")[0])

        # Find matching AI image (positive)
        positive = None

        for gen in generated:

            gen_num = int(gen.split("_")[1].split(" ")[0])

            if gen_num == org_num:
                positive = gen
                break

        if positive is None:
            continue

        positive_path = os.path.join(
            generated_dir,
            positive
        )

        # Every incorrect AI image becomes a negative
        for gen in generated:

            gen_num = int(gen.split("_")[1].split(" ")[0])

            if gen_num == org_num:
                continue

            negative_path = os.path.join(
                generated_dir,
                gen
            )

            triplets.append([
                os.path.join(original_dir, org),
                positive_path,
                negative_path
            ])

    random.shuffle(triplets)

    df = pd.DataFrame(
        triplets,
        columns=[
            "anchor",
            "positive",
            "negative"
        ]
    )

    df.to_csv(output_csv, index=False)

    print(f"\n{output_csv} created")
    print(df.head())
    print("Total Triplets :", len(df))


generate_triplets(
    "dataset/train_original",
    "dataset/train_generated",
    "dataset/train_triplets.csv"
)

generate_triplets(
    "dataset/test_original",
    "dataset/test_generated",
    "dataset/test_triplets.csv"
)