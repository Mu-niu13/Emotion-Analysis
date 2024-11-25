# lib/evaluate.py

import torch
import torch.nn as nn
from tqdm import tqdm

def eval_model(model, data_loader, loss_fn, device, n_examples, n_mc_samples=10):
    model = model.eval()
    losses = []
    all_outputs = []
    all_labels = []
    uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Collect outputs over multiple forward passes
            mc_outputs = []
            for _ in range(n_mc_samples):
                # Enable dropout during evaluation
                model.train()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                mc_outputs.append(outputs.unsqueeze(2))  # Shape: (batch_size, n_classes, 1)
            
            # Stack outputs
            mc_outputs = torch.cat(mc_outputs, dim=2)  # Shape: (batch_size, n_classes, n_mc_samples)
            # Compute mean and variance
            mean_outputs = mc_outputs.mean(dim=2)
            var_outputs = mc_outputs.var(dim=2)
            
            loss = loss_fn(mean_outputs, labels)
            losses.append(loss.item())
            
            all_outputs.append(mean_outputs.cpu())
            all_labels.append(labels.cpu())
            uncertainties.append(var_outputs.cpu())
            
    return np.mean(losses), torch.cat(all_outputs), torch.cat(all_labels), torch.cat(uncertainties)
