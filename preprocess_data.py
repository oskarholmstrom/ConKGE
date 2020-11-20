import networkx as nx
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def preprocess_data(graph, ent_vocab, rel_vocab, batch_size, eval_set=False, rand=False):

    if eval_set:
        # NOTE: The test pre-processing is made for FB15k, where a triple appears twice as ([MASK], r, o) and (s, r, [MASK])
        orig_inputs, inputs, positions, attention_masks, labels = all_triples_test(graph, ent_vocab, rel_vocab)
    else:
        #orig_inputs, inputs, positions, attention_masks, labels = all_triples_training(graph, ent_vocab, rel_vocab)
        orig_inputs, inputs, positions, attention_masks, labels = all_triples_test(graph, ent_vocab, rel_vocab)

    dataloader = create_dataloader(orig_inputs, inputs, positions, attention_masks, labels, batch_size, rand)

    return dataloader

def all_triples_test(graph, ent_vocab, rel_vocab):
    all_inputs = []
    orig_inputs = []
    attention_masks = []
    all_positions = []
    all_labels = []
    for triple in graph.get_triples():

        orig_input = [ent_vocab['[CLS]'], ent_vocab[triple[0]], ent_vocab[triple[1]], rel_vocab[triple[2][0]]]
        orig_inputs.append(orig_input)
        orig_inputs.append(orig_input)


        # Head mask
        input_triple = [ent_vocab['[CLS]'], ent_vocab['[MASK]'], ent_vocab[triple[1]], rel_vocab[triple[2][0]]]
        all_inputs.append(input_triple)

        # Tail mask
        input_triple = [ent_vocab['[CLS]'], ent_vocab[triple[0]], ent_vocab['[MASK]'], rel_vocab[triple[2][0]]]
        all_inputs.append(input_triple)

        label_triple_head = [-100, ent_vocab[triple[0]], -100, -100]
        all_labels.append(label_triple_head)
        label_triple_tail = [-100, -100, ent_vocab[triple[1]], -100]
        all_labels.append(label_triple_tail)

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

    return orig_inputs, all_inputs, all_positions, attention_masks, all_labels


def all_triples_training(graph, ent_vocab, rel_vocab):

    all_inputs = []
    orig_inputs = []
    attention_masks = []
    all_positions = []
    all_labels = []
    for triple in graph.get_triples():
        orig_inputs.append([ent_vocab['[CLS]'], ent_vocab[triple[0]], ent_vocab[triple[1]], rel_vocab[triple[2][0]]])

        encoded_triple = [ent_vocab[triple[0]], rel_vocab[triple[2][0]], ent_vocab[triple[1]]]
        masked_triple, label_triple = masking_scheme(graph, ent_vocab, rel_vocab, encoded_triple)
        input_seq = [ent_vocab['[CLS]'], masked_triple[0], masked_triple[2], masked_triple[1]]
        all_inputs.append(input_seq)

        # NOTE: Only CLS is labels as not to be predicted upon.
        #       Possibly everything but the masked token(s) should be labeled -100
        label_seq = [-100, label_triple[0], label_triple[2], label_triple[1]]
        all_labels.append(label_seq)

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

    return orig_inputs, all_inputs, all_positions, attention_masks, all_labels



def masking_scheme(graph, ent_vocab, rel_vocab, triple):
    """
    15% of all tokens are selected for prediction:
        - 12% are replaced with [MASK]
        - 1.5% are replaced with a random entity/relation token
        - 1.5% are kept intact
    """
    # NOTE: Current implementation only takes triples into account and only mask at most one token per triple
    mask_choice = np.random.randint(0, graph.num_triples)
    pos = np.random.randint(0, 3)
    masked_triple = triple
    label_triple = [-100, -100, -100]

    if mask_choice <= round(graph.num_triples*0.15):

        # Token left intact iff mask_choice <= num_triple*0.15 and > num_triples*0.135

        # Mask the token
        if mask_choice <= round(graph.num_triples*0.135):
            masked_triple[pos] = ent_vocab['[MASK]']

        # Replace with random token
        elif mask_choice <= round(graph.num_triples*0.015):
            if pos % 2 == 0:
                random_token = ent_vocab[np.random.choice(list(ent_vocab))]
            else:
                random_token = rel_vocab[np.random.choice(list(rel_vocab))]
            masked_triple[pos] = random_token

        # Set label such that masked token will be predicted
        label_triple[pos] = triple[pos]

    return masked_triple, label_triple




def create_dataloader(orig_inputs, inputs, positions, masks, labels, batch_size, rand=False):
    """
    Converts the inputs, positions, masks and labels into tensors,
    create a dataset and DataLoader with either a RandomSampler
    or SequentialSampler, depending if an argument to rand
    is True. Returns a PyTorch DataLoader object.
    """
    tensor_orig_inputs = torch.LongTensor(orig_inputs)
    tensor_inputs = torch.LongTensor(inputs)
    tensor_positions = torch.LongTensor(positions)
    tensor_masks = torch.LongTensor(masks)
    tensor_labels = torch.LongTensor(labels)

    dataset = TensorDataset(tensor_orig_inputs, tensor_inputs, tensor_positions, tensor_masks, tensor_labels)

    if rand == True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
