# data/prepare.py
import os, pandas as pd

def build_csv(split="test"):
    base = f"data/chest_xray/{split}"
    rows = []
    for label_name, label_val in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        folder = os.path.join(base, label_name)
        for fname in os.listdir(folder):
            if fname.endswith(".jpeg") or fname.endswith(".jpg"):
                rows.append({
                    "filepath": os.path.join(folder, fname),
                    "label": label_val,
                    "label_name": label_name
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"data/{split}_labels.csv", index=False)
    print(f"{split}: {len(df)} images")
    print(df["label_name"].value_counts())
    return df

build_csv("test")