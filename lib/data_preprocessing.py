import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label_columns):
        self.tokenizer = tokenizer
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe[label_columns].values
        self.max_len = max_len
        self.label_columns = label_columns
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        label_vector = self.labels[index]
        label_indices = np.where(label_vector == 1)[0]
        label_index = label_indices[0]
        labels = torch.tensor(label_index, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': text
        }

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def get_label_columns(df):
    label_columns = df.columns.tolist()
    label_columns.remove('id')
    label_columns.remove('text')
    label_columns.remove('example_very_unclear')
    return label_columns

def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == 'text':
            batch_dict[key] = [d[key] for d in batch]
        else:
            batch_dict[key] = torch.stack([d[key] for d in batch], dim=0)
    return batch_dict

def create_data_loader(df, tokenizer, max_len, batch_size, label_columns):
    ds = EmotionDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_len=max_len,
        label_columns=label_columns
    )
    
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
