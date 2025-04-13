import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone
import dateutil.parser

TASK = """Classify this tweet into one of these humanitarian categories:

1. **Caution and advice** - Warnings, safety tips, or official guidance
2. **Displaced people** - Evacuations, shelters, missing family reports
3. **Infrastructure damage** - Collapsed buildings, flooded roads, power outages
4. **Injured/dead** - Casualty reports, injury details, body recovery
5. **Requests/needs** - Urgent asks for food, medicine, or rescue
6. **Rescue efforts** - Volunteer activities, donation drives
7. **Not humanitarian** - Irrelevant tweets, casual conversation
8. **Other** - Humanitarian but doesn't fit other categories

"""


class DisasterDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, label_encoder=None, desc_csv_path=None, prompt_heuristic=False):
        self.data = pd.read_csv(data_path, sep="\t")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_heuristic = prompt_heuristic
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.data["class_label"].unique())
        else:
            self.label_encoder = label_encoder
            unique_labels = set(self.data["class_label"].unique())
            encoder_classes = set(self.label_encoder.classes_)
            if not unique_labels.issubset(encoder_classes):
                raise ValueError(f"Found labels {unique_labels - encoder_classes} not present in the label encoder")
        if desc_csv_path is not None:
            desc_df = pd.read_csv(desc_csv_path, sep="|")
            base = os.path.basename(data_path)
            event_name = base.rsplit('_', 1)[0]
            event_row = desc_df[desc_df["dataset name"] == event_name]
            if not event_row.empty:
                self.event_time = dateutil.parser.parse(event_row.iloc[0]["event time"]).replace(tzinfo=timezone.utc)
                self.event_desc = event_row.iloc[0]["description of what happened"]
            else:
                self.event_time = None
                self.event_desc = ""
        else:
            self.event_time = None
            self.event_desc = ""

    def __len__(self):
        return len(self.data)
    
    def tweet_id_to_datetime(self, tweet_id):
        twitter_epoch = 1288834974657
        timestamp_ms = (int(tweet_id) >> 22) + twitter_epoch
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return dt
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["tweet_text"]
        tweet_time = self.tweet_id_to_datetime(row["tweet_id"])
        label = self.label_encoder.transform([row["class_label"]])[0]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        if self.event_time is not None:
            delta = tweet_time - self.event_time
            days_since_event = delta.days
        else:
            days_since_event = "N/A"


        if self.prompt_heuristic:
            formatted_text = ("The tweets time since event happened : {} days\n\n"
                            "Description of the event : {}\n\n"
                            "The tweet to classify: {}").format(days_since_event, self.event_desc, text)

            formatted_text = TASK + "\n\n" + formatted_text
        else:
            formatted_text = text


        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label),
            "formatted_text": formatted_text
        }

