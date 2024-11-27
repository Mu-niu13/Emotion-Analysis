import torch
import torch.nn as nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3, model_name='distilbert-base-uncased'):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, n_classes)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs[0]
        pooled_output = torch.mean(last_hidden_state, dim=1)
        
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
