from PIL import Image
import pandas as pd

df = pd.read_csv("dataset/train_triplets.csv")

print("Rows:", len(df))

for i, row in df.iterrows():

    print(f"Checking row {i}...", end=" ")

    try:

        print("Anchor", end=" ")
        Image.open(row["anchor"]).load()

        print("Positive", end=" ")
        Image.open(row["positive"]).load()

        print("Negative", end=" ")
        Image.open(row["negative"]).load()

        print("OK")

    except Exception as e:

        print("\nFAILED")

        print(row)

        print(e)

        break