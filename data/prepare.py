import os, pandas as pd

def build_csv(split="test"):
    base = f"data/chest_xray/{split}"
    rows = []
    for label_name, label_val in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        folder = f"{base}/{label_name}"          # forward slash hardcoded
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                rows.append({
                    "filepath": f"{folder}/{fname}",   # forward slash
                    "label": label_val,
                    "label_name": label_name
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"data/{split}_labels.csv", index=False)
    print(f"Saved {len(df)} rows")
    print(df["filepath"].iloc[0])
    return df

build_csv("test")