# data/dataset.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CXRDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        tensor = self.transform(img)
        # unique id = just the filename without extension
        fname_id = os.path.basename(
            row["filepath"]).replace(".jpeg","").replace(".jpg","")
        return tensor, int(row["label"]), fname_id