from PIL import Image
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("dataset/train_triplets.csv")

print("Rows:", len(df))

for i, row in tqdm(df.iterrows(), total=len(df), desc="Checking dataset"):

    try:

        Image.open(row["anchor"].replace('\\', '/')).load()
        Image.open(row["positive"].replace('\\', '/')).load()
        Image.open(row["negative"].replace('\\', '/')).load()

    except Exception as e:

        print(f"\nFAILED on row {i}")

        print(row)

        print(e)

        break