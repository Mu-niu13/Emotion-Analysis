import torch
import torch.nn as nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3, model_name='bert-base-uncased'):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        # classifier head
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.model.config.hidden_size, 512)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        
        # init weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        x = self.layer_norm(pooled_output)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
