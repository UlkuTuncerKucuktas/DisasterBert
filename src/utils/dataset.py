import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class DisasterDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, label_encoder=None):
        self.data = pd.read_csv(data_path, sep="\t")
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Handle label encoder
        if label_encoder is None:
            # If no label encoder provided, create and fit a new one
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.data["class_label"].unique())
        else:
            # Use the provided label encoder
            self.label_encoder = label_encoder
            # Verify all labels in this dataset are in the encoder's classes
            unique_labels = set(self.data["class_label"].unique())
            encoder_classes = set(self.label_encoder.classes_)
            if not unique_labels.issubset(encoder_classes):
                raise ValueError(f"Found labels {unique_labels - encoder_classes} not present in the label encoder")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["tweet_text"]
        label = self.label_encoder.transform([row["class_label"]])[0]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label)
        }
