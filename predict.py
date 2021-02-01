import torch
import numpy as np
from model import ConKGE

def predict(args, model, test_dataloader, device):
    """
    Computes the predictions for a masked node.
    Returns the predictions, and the inputs and labels that where predicted on.
    """

    model.eval()

    preds = []
    true_inputs = []
    true_labels = []
    count = 0
    for batch in test_dataloader:
        count+=1
        batch = tuple(t.to(device) for t in batch)
        
        orig_inputs = batch[0]
        inputs = batch[1]
        positions = batch[2]
        masks = batch[3]
        labels = batch[4]


        with torch.no_grad():
            outputs = model(args=args,
                            input_ids = inputs,
                            attention_mask=masks,
                            position_ids = positions
                            )
       
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        orig_labels = labels.to('cpu').numpy()
        orig_inputs = orig_inputs.to('cpu').numpy()

        preds.extend(logits)
        true_inputs.extend(orig_inputs)    # Save the inputs in order find which token was masked
        true_labels.extend(orig_labels)    # Save the labels in order find the original triple

    return preds, true_inputs, true_labels
