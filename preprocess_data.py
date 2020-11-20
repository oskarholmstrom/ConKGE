import networkx as nx
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def preprocess_data(graph, ent_vocab, rel_vocab, batch_size, eval_set=False, rand=False):

    if eval_set:
        # NOTE: The test pre-processing is made for FB15k, where a triple appears twice as ([MASK], r, o) and (s, r, [MASK])
        inputs, positions, attention_masks, labels = all_triples_test(graph, ent_vocab, rel_vocab)
    else:
        inputs, positions, attention_masks, labels = all_triples_training(graph, ent_vocab, rel_vocab)

    dataloader = create_dataloader(inputs, positions, attention_masks, labels, batch_size, rand)

    return dataloader

def all_triples_test(graph, ent_vocab, rel_vocab):
    all_inputs = []
    attention_masks = []
    all_positions = []
    all_labels = []
    for triple in graph.get_triples():

        # Head mask
        input_triple = [ent_vocab['[CLS]'], ent_vocab['[MASK]'], ent_vocab[triple[1]], rel_vocab[triple[2][0]]]
        all_inputs.append(input_triple)

        # Tail mask
        input_triple = [ent_vocab['[CLS]'], ent_vocab[triple[0]], ent_vocab['[MASK]'], rel_vocab[triple[2][0]]]
        all_inputs.append(input_triple)

        label_triple = [-100, ent_vocab[triple[0]], ent_vocab[triple[1]], rel_vocab[triple[2][0]]]
        all_labels.append(label_triple)
        all_labels.append(label_triple)

        all_positions.append([0,1,3,2])
        all_positions.append([0,1,3,2])

        # Create a unique attention mask for each token, where they can only
        # attend to adjacent entity nodes and the relation between them
        adj_matrix = [
                        [1, 1, 1, 1],
                        [0, 1, 0, 1],
                        [0, 0, 1, 1],
                        [0, 1, 1, 1],
                     ]
        attention_masks.append(adj_matrix)
        attention_masks.append(adj_matrix)

    return all_inputs, all_positions, attention_masks, all_labels


def all_triples_training(graph, ent_vocab, rel_vocab):

    all_inputs = []
    attention_masks = []
    all_positions = []
    all_labels = []
    for triple in graph.get_triples():

        masked_triple = masking_scheme(graph, ent_vocab, rel_vocab, [triple[0], triple[2][0], triple[1]])
        input_triple = [ent_vocab['[CLS]'], ent_vocab[masked_triple[0]], ent_vocab[masked_triple[2]], rel_vocab[masked_triple[1]]]
        all_inputs.append(input_triple)

        # NOTE: Only CLS is labels as not to be predicted upon. 
        #       Possibly everything but the masked token(s) should be labeled -100
        label_triple = [-100, ent_vocab[triple[0]], ent_vocab[triple[1]], rel_vocab[triple[2][0]]]
        all_labels.append(label_triple)

        # NOTE: Hard-coded for triples
        all_positions.append([0,1,3,2])

        # Create a unique attention mask for each token, where they can only
        # attend to adjacent entity nodes and the relation between them

        # NOTE: Hard-coded for triples
        adj_matrix = [
                        [1, 1, 1, 1],
                        [0, 1, 0, 1],
                        [0, 0, 1, 1],
                        [0, 1, 1, 1],
                     ]
        attention_masks.append(adj_matrix)

    return all_inputs, all_positions, attention_masks, all_labels



def masking_scheme(graph, ent_vocab, rel_vocab, triple):
    """
    15% of all tokens are selected for prediction:
        - 12% are replaced with [MASK]
        - 1.5% are replaced with a random entity/relation token
        - 1.5% are kept intact
    """
    # NOTE: Current implementation do not keep 1.5% intact.

    mask_choice = np.random.randint(0, graph.num_triples)
    pos = np.random.randint(0, 3)
    masked_triple = triple

    if mask_choice <= round(graph.num_triples*0.12):
        if mask_choice <= round(graph.num_triples*0.015):
            if pos % 2 == 0:
                random_token = np.random.choice(list(ent_vocab))
            else:
                random_token = np.random.choice(list(rel_vocab))
            masked_triple[pos] = random_token
        else:
            masked_triple[pos] = '[MASK]'
            
    return masked_triple




def create_dataloader(inputs, positions, masks, labels, batch_size, rand=False):
    """
    Converts the inputs, positions, masks and labels into tensors,
    create a dataset and DataLoader with either a RandomSampler
    or SequentialSampler, depending if an argument to rand
    is True. Returns a PyTorch DataLoader object.
    """

    tensor_inputs = torch.LongTensor(inputs)
    tensor_positions = torch.LongTensor(positions)
    tensor_masks = torch.LongTensor(masks)
    tensor_labels = torch.LongTensor(labels)

    dataset = TensorDataset(tensor_inputs, tensor_positions, tensor_masks, tensor_labels)

    if rand == True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
