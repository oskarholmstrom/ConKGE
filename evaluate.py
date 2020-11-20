import torch
import numpy as np
from model import ConKGE

def predict(model, test_dataloader, device):
    
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
            outputs = model(input_ids = inputs,
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


def hits(preds, true_inputs, true_labels, dataset):
    """
    For each predicition it is evaluated if the true label is in the
    top k predictions, for k = 1, 3, 10.
    It is done in a filtered settings, so for all the predictions that
    are valid (triples exist in any dataset), and are not the true label,
    are removed.
    """
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    total = 0
    for pred, true_input, true_label in zip(preds, true_inputs, true_labels):

        for i, token in enumerate(true_label):

            if token.item() != -100: # Check if a token in the input as masked
                total += 1

                # Remove all predictions that are correct (part of a triple) but are not the true label
                filtered_pred = remove_correct_preds(pred[i], true_input, i, dataset)

                # Sort in ascending order and retrieve the original indices
                sorted_pred = np.argsort(filtered_pred)
                true_token_id = true_label[i].item()

                if true_token_id in sorted_pred[::-1][:1]:
                    hits_1 += 1
                if true_token_id in sorted_pred[::-1][:3]:
                    hits_3 += 1
                if true_token_id in sorted_pred[::-1][:10]:
                    hits_10 += 1

    return hits_1, hits_3, hits_10, total, hits_1/total, hits_3/total, hits_10/total

def remove_correct_preds(pred, true_triple, ind, dataset):

    # Head is masked
    if ind == 1:
        tail_rel = str(true_triple[2].item()) + ', ' + str(true_triple[3].item())
        adj_nodes = list(dataset.tail_rel[tail_rel])

    # Tail is masked
    if ind == 2:
        head_rel = str(true_triple[1].item()) + ', ' + str(true_triple[3].item())
        adj_nodes = list(dataset.head_rel[head_rel])

    adj_nodes.remove(true_triple[ind].item())

    filtered_pred = np.delete(pred, adj_nodes)

    return filtered_pred