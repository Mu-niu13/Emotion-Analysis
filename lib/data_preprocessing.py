# lib/data_preprocessing.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label_columns):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['text']
        self.labels = dataframe[label_columns].values
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = torch.tensor(self.labels[index], dtype=torch.float)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
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
        num_workers=2
    )
