import numpy as np

def evaluate(preds, true_inputs, true_labels, dataset):
    """
    For each predicition it is evaluated if the true label is in the
    top k predictions, for k = 1, 3, 10, and the MRR is calculated.
    The evaluation is done in a filtered setting. For all the predictions that
    are valid (triples exist in any dataset), and are not the true label,
    are removed from the ranked predictions.
    """

    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    sum_reciprocal_rank = 0
    total = 0

    for pred, true_input, true_label in zip(preds, true_inputs, true_labels):
        for i, token in enumerate(true_label):

            if token.item() != -100: # Check if a token in the input as masked

                # Sort in ascending order and retrieve the original indices
                sorted_pred = np.argsort(pred[i])

                # Remove all predictions that create real triples in the dataset (except the triple to be predicted)
                filtered_pred = remove_correct_preds(sorted_pred, true_input, i, dataset)

                true_token_id = true_label[i].item()

                rank = np.where(filtered_pred == true_token_id)
                rank = len(filtered_pred)-rank[0]
                sum_reciprocal_rank += 1/rank[0]
                if true_token_id in filtered_pred[::-1][:1]:
                    hits_1 += 1
                if true_token_id in filtered_pred[::-1][:3]:
                    hits_3 += 1
                if true_token_id in filtered_pred[::-1][:10]:
                    hits_10 += 1

                total += 1

    return sum_reciprocal_rank/total, hits_1, hits_3, hits_10, hits_1/total, hits_3/total, hits_10/total

def remove_correct_preds(pred, true_triple, ind, dataset):
    """
    Removes predictions that exist in the dataset, to create
    a filtered setting for the evaluation.
    """

    # Head is masked => Get all true triples with the corresponding rel and tail entity.
    if ind == 1:
        tail_rel = str(true_triple[2].item()) + ', ' + str(true_triple[3].item())
        adj_nodes = list(dataset.tail_rel[tail_rel])

    # Tail is masked => Get all true triples with the corresponding head and rel entity.
    if ind == 2:
        head_rel = str(true_triple[1].item()) + ', ' + str(true_triple[3].item())
        adj_nodes = list(dataset.head_rel[head_rel])

    # Removes the true triple from the triples that will be filtered
    adj_nodes.remove(true_triple[ind].item())

    
    remove_ind = []
    for node in adj_nodes:
        remove_ind.append(np.where(pred == node))

    #Removes all triples that are in the dataset from the ranked predictions
    filtered_pred = np.delete(pred, remove_ind)

    return filtered_pred