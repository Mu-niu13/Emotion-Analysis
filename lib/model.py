# lib/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3, model_name='distilbert-base-uncased'):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # For DistilBERT, use outputs[0][:, 0, :] as the pooled output
        pooled_output = outputs[0][:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)
