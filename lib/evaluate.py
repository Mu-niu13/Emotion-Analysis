import torch
from tqdm import tqdm
import numpy as np

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    return np.mean(losses), all_outputs, all_labels