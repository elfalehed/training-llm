import torch
from torch.nn import CrossEntropyLoss
import numpy as np

def calculate_perplexity(model, eval_dataset, device):
    model.eval()
    loss_fn = CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_dataset:
            inputs = batch["input_ids"].to(device)
            outputs = model(inputs)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity
