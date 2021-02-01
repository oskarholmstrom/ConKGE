import networkx as nx
import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def preprocess_train_data(args, graph, element_vocab, max_seq_len, batch_size, rand=True):

    orig_inputs = []
    inputs = []
    positions = []
    attention_masks = []
    labels = []

    # Adds all triples twice, with the head or tail masked in each.
    if args.triples_head_tail:
        add_triples_entity_mask(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len)
        print("Triples ent  added. No training samples: ", len(orig_inputs))

    # Add triples with the relation node masked to training data
    if args.triples_rel:
        add_triples_rel_mask(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len)
        print("Triples rel added. No training samples: ", len(orig_inputs))

    # Adds paths of length=path_length to training data
    if args.path_length != 0:
        for i in range(0, args.path_length):
            add_paths(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len, i+1, args.num_rand_paths)
            print("Paths added. No training samples: ", len(orig_inputs))
    
    # Add graphs of connected triples to the training data
    if args.subgraph_triples:
        add_subgraph_triples(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len, args.num_rand_paths, args.num_samples)
        print("Subgraph triples added. No training samples: ", len(orig_inputs))

    dataloader = create_dataloader(orig_inputs, inputs, positions, attention_masks, labels, batch_size, rand)


    return dataloader

def preprocess_test_data(args, graph, element_vocab, batch_size, rand=False):

    orig_inputs = []
    inputs = []
    positions = []
    attention_masks = []
    labels = []
    #orig_inputs, inputs, positions, attention_masks, labels = add_triples_test(args, graph, element_vocab, args.max_seq_len)

    add_triples_entity_mask(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, args.max_seq_len)

    dataloader = create_dataloader(orig_inputs, inputs, positions, attention_masks, labels, batch_size, rand)

    return dataloader

def add_triples_test(args, graph, element_vocab, max_seq_len):

    all_inputs = []
    orig_inputs = []
    attention_masks = []
    all_positions = []
    all_labels = []


    for triple in graph.get_triples():

        # Encoding for the triple (triple restructured as head, tail rel)
        orig_input = [element_vocab['[CLS]'], element_vocab[triple[0]], element_vocab[triple[1]], element_vocab[triple[2][0]]]
        
        orig_inputs.append(orig_input)
        orig_inputs.append(orig_input)

        # Head node masked
        input_triple = [element_vocab['[CLS]'], element_vocab['[MASK]'], element_vocab[triple[1]], element_vocab[triple[2][0]]]
        all_inputs.append(input_triple)

        # Tail node masked
        input_triple = [element_vocab['[CLS]'], element_vocab[triple[0]], element_vocab['[MASK]'], element_vocab[triple[2][0]]]
        all_inputs.append(input_triple)

        # All nodes that are not to be predicted is labeled with -100
        label_triple_head = [-100, element_vocab[triple[0]], -100, -100]
        all_labels.append(label_triple_head)
        label_triple_tail = [-100, -100, element_vocab[triple[1]], -100]
        all_labels.append(label_triple_tail)

        all_positions.append([0,1,3,2])
        all_positions.append([0,1,3,2])


        if args.node_specific_mask:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        elif args.node_specific_mask_directed:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        
        elif args.node_specific_mask_ent_rel:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)

        elif args.node_specific_mask_ent_rel_directed:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        else:
            adj_matrix = [1, 1, 1, 1]
            pad_seq(adj_matrix, max_seq_len, 0)
        attention_masks.append(adj_matrix)
        attention_masks.append(adj_matrix)

    return orig_inputs, all_inputs, all_positions, attention_masks, all_labels


def add_triples_entity_mask(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len):

    for triple in graph.get_triples():

        orig_input = [element_vocab['[CLS]'], element_vocab[triple[0]], element_vocab[triple[1]], element_vocab[triple[2][0]]]
        pad_seq(orig_input, max_seq_len, element_vocab['[PAD]'])
        orig_inputs.append(orig_input)
        orig_inputs.append(orig_input)

        # Head mask
        input_triple = [element_vocab['[CLS]'], element_vocab['[MASK]'], element_vocab[triple[1]], element_vocab[triple[2][0]]]
        pad_seq(input_triple, max_seq_len, element_vocab['[PAD]'])
        inputs.append(input_triple)

        # Tail mask
        input_triple = [element_vocab['[CLS]'], element_vocab[triple[0]], element_vocab['[MASK]'], element_vocab[triple[2][0]]]
        pad_seq(input_triple, max_seq_len, element_vocab['[PAD]'])
        inputs.append(input_triple)

        label_triple_head = [-100, element_vocab[triple[0]], -100, -100]
        pad_seq(label_triple_head, max_seq_len, -100)
        labels.append(label_triple_head)

        label_triple_tail = [-100, -100, element_vocab[triple[1]], -100]
        pad_seq(label_triple_tail, max_seq_len, -100)
        labels.append(label_triple_tail)

        pos = [0, 1, 3, 2]
        pad_seq(pos, max_seq_len, 0)
        positions.append(pos)
        positions.append(pos)

        if args.node_specific_mask:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1]
                         ]
            pad_att = [0]*4
            
            # Append a row for padded tokens
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            # Pad each row
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        elif args.node_specific_mask_directed:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        
        elif args.node_specific_mask_ent_rel:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)

        elif args.node_specific_mask_ent_rel_directed:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        else:
            adj_matrix = [1, 1, 1, 1]
            pad_seq(adj_matrix, max_seq_len, 0)

        attention_masks.append(adj_matrix)
        attention_masks.append(adj_matrix)


def add_triples_rel_mask(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len):

    for triple in graph.get_triples():
 
        orig_input = [element_vocab['[CLS]'], element_vocab[triple[0]], element_vocab[triple[1]], element_vocab[triple[2][0]]]
        pad_seq(orig_input, max_seq_len, element_vocab['[PAD]'])
        orig_inputs.append(orig_input)

        # Rel mask
        input_triple = [element_vocab['[CLS]'], element_vocab[triple[0]], element_vocab[triple[1]], element_vocab['[MASK]']]
        pad_seq(input_triple, max_seq_len, element_vocab['[PAD]'])
        inputs.append(input_triple)

        label_triple_head = [-100, -100, -100, element_vocab[triple[2][0]]]
        pad_seq(label_triple_head, max_seq_len, -100)
        labels.append(label_triple_head)

        pos = [0, 1, 3, 2]
        pad_seq(pos, max_seq_len, 0)
        positions.append(pos)

        if args.node_specific_mask:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 1, 1, 1]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        elif args.node_specific_mask_directed:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        
        elif args.node_specific_mask_ent_rel:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)

        elif args.node_specific_mask_ent_rel_directed:
            adj_matrix = [
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]
                         ]
            pad_att = [0]*4
            for i in range(0, max_seq_len-len(adj_matrix)):
                adj_matrix.append(pad_att)
            pad_seq(adj_matrix, max_seq_len, 0, two_dim=True)
        else:
            adj_matrix = [1, 1, 1, 1]
            pad_seq(adj_matrix, max_seq_len, 0)


        attention_masks.append(adj_matrix)


def add_paths(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len, path_len, no_paths):
    # For each triple, add no_paths number of paths of k length.
    # Where k=1: h->r->h'->r->t ...

    for triple in graph.get_triples():

        triple_ordered = [triple[0], triple[2][0], triple[1]]
        triple_tail = triple_ordered[-1]

        # Get all triples in which the tail is the head of.
        tail_neighbors = graph.get_neighbors(triple_tail)

        if len(tail_neighbors) >= no_paths:
            # If there are more triples than the number of paths to be selected
            # select k random number of paths
            random_select = random.choices(tail_neighbors, k=no_paths)

        elif len(tail_neighbors) == 0:
            # If there are no such triples, create paths for next triple in the dataset.
            continue
        else:
            random_select = random.choices(tail_neighbors, k=len(tail_neighbors))

        all_paths = []
        for ent_rel in random_select:
            all_paths.append(triple_ordered + [ent_rel[1], ent_rel[0]])

        # If paths are longer than path_length=1, then repeat the random selection to extend the path
        k = path_len-1
        while k > 0:
            new_paths = []
            for path in all_paths:

                tail = path[-1]
                tail_neighbors = graph.get_neighbors(tail)

                if len(tail_neighbors) > 0:
                    random_neighbor = random.choice(tail_neighbors)
                    new_paths.append(path + [random_neighbor[1], random_neighbor[0]])
            k -= 1
            all_paths = new_paths

        for path in all_paths:
            path_seq = [element_vocab['[CLS]']]
            num_ents = 0
            num_rels = 0
            pos = [0]
            for i, token in enumerate(path):
                if i % 2 == 0:
                    path_seq.append(element_vocab[token])
                    num_ents +=1
                    pos.append(i+1)
            for i, token in enumerate(path):
                if i % 2 != 0:
                    path_seq.append(element_vocab[token])
                    num_rels +=1
                    pos.append(i+1)

            pad_seq(path_seq, max_seq_len, element_vocab['[PAD]'])
            orig_inputs.append(path_seq)
            orig_inputs.append(path_seq)

            pad_seq(pos, max_seq_len, 0)
            positions.append(pos)
            positions.append(pos)

            if args.node_specific_mask:
                att_mask = []
                cls_mask = [1]*(num_ents+num_rels+1)
                pad_seq(cls_mask, max_seq_len, 0)
                att_mask.append(cls_mask)

                att_mask_node = [0]*len(path_seq)
                pos_dict = {}
                for i, p in enumerate(pos):
                    if p != 0:
                        pos_dict[p] = i

                for i, p in enumerate(pos):
                    att_seq = [0]*len(path_seq)
                    if p != 0:
                        # If the token is not CLS or PAD
                        att_seq[pos_dict[p]] = 1
                        if p-1 in pos_dict:
                            att_seq[pos_dict[p-1]] = 1
                        if p+1 in pos_dict:
                            att_seq[pos_dict[p+1]] = 1
                        att_mask.append(att_seq)
                    if i > 0 and p == 0:
                        # If the token is PAD
                        att_mask.append([0]*len(path_seq))

            elif args.node_specific_mask_directed:
                att_mask = []
                cls_mask = [1]*(num_ents+num_rels+1)
                pad_seq(cls_mask, max_seq_len, 0)
                att_mask.append(cls_mask)

                att_mask_node = [0]*len(path_seq)
                pos_dict = {}
                for i, p in enumerate(pos):
                    if p != 0:
                        pos_dict[p] = i

                for i, p in enumerate(pos):
                    att_seq = [0]*len(path_seq)
                    if p != 0:
                        att_seq[pos_dict[p]] = 1
                        if p+1 in pos_dict:
                            att_seq[pos_dict[p+1]] = 1
                        att_mask.append(att_seq)
                    if i > 0 and p == 0:
                        att_mask.append([0]*len(path_seq))

            elif args.node_specific_mask_ent_rel:
                att_mask = []
                cls_mask = [1]*(num_ents+num_rels+1)
                pad_seq(cls_mask, max_seq_len, 0)
                att_mask.append(cls_mask)

                att_mask_node = [0]*len(path_seq)
                pos_dict = {}
                for i, p in enumerate(pos):
                    if p != 0:
                        pos_dict[p] = i

                for i, p in enumerate(pos):
                    att_seq = [0]*len(path_seq)
                    if p != 0 and p%2 != 0:
                        # Mask an entity
                        att_seq[pos_dict[p]] = 1
                        if p-1 in pos_dict:
                            att_seq[pos_dict[p-1]] = 1
                            att_seq[pos_dict[p-2]] = 1
                        if p+1 in pos_dict:
                            att_seq[pos_dict[p+1]] = 1
                            att_seq[pos_dict[p+2]] = 1
                        att_mask.append(att_seq)

                    elif p != 0 and p%2 == 0:
                        # Mask the relation
                        att_seq[pos_dict[p]] = 1
                        if p-1 in pos_dict:
                            att_seq[pos_dict[p-1]] = 1
                        if p+1 in pos_dict:
                            att_seq[pos_dict[p+1]] = 1
                        att_mask.append(att_seq)

                    if i > 0 and p == 0:
                        att_mask.append([0]*len(path_seq))
                

            elif args.node_specific_mask_ent_rel_directed:
                att_mask = []
                cls_mask = [1]*(num_ents+num_rels+1)
                pad_seq(cls_mask, max_seq_len, 0)
                att_mask.append(cls_mask)

                att_mask_node = [0]*len(path_seq)
                pos_dict = {}
                for i, p in enumerate(pos):
                    if p != 0:
                        pos_dict[p] = i

                for i, p in enumerate(pos):
                    att_seq = [0]*len(path_seq)
                    if p != 0 and p%2 != 0:
                        # Mask an entity
                        att_seq[pos_dict[p]] = 1
                        if p+1 in pos_dict:
                            att_seq[pos_dict[p+1]] = 1
                            att_seq[pos_dict[p+2]] = 1
                        att_mask.append(att_seq)
                    elif p != 0 and p%2 == 0:
                        # Mask the relation
                        att_seq[pos_dict[p]] = 1
                        if p+1 in pos_dict:
                            att_seq[pos_dict[p+1]] = 1
                        att_mask.append(att_seq)
                        
                    if i > 0 and p == 0:
                        att_seq = [0]*len(path_seq)
                        att_mask.append(att_seq)
            else:
                att_mask = [1]*(num_ents+num_rels+1)
                pad_seq(att_mask, max_seq_len, 0)

            attention_masks.append(att_mask)
            attention_masks.append(att_mask)
            mask_head = path_seq[:]
            mask_head[1] = element_vocab['[MASK]']
            inputs.append(mask_head)

            mask_tail = path_seq[:]
            mask_tail[num_ents] = element_vocab['[MASK]']
            inputs.append(mask_tail)

            labels_head = [-100]*len(path_seq)
            labels_tail = [-100]*len(path_seq)
            labels_head[1] = path_seq[1]
            labels_tail[num_ents] = path_seq[num_ents]

            labels.append(labels_head)
            labels.append(labels_tail)



def add_subgraph_triples(args, orig_inputs, inputs, positions, attention_masks, labels, graph, element_vocab, max_seq_len, no_paths, no_samples):

    for node in graph.get_nodes():
        tail_neighbors = graph.get_neighbors(node)

        # If a node has no neighboring nodes, go to next node
        if len(tail_neighbors) == 0:
            continue

        # Create a graph n=no_samples times
        for i in range(0, no_samples):

            if len(tail_neighbors) >= no_paths:
                random_select = random.choices(tail_neighbors, k=no_paths)

            else:
                random_select = random.choices(tail_neighbors, k=len(tail_neighbors))


            path_seq = [element_vocab['[CLS]'], element_vocab[node]]

            pos = [0, 1]

            if args.node_specific_mask or args.node_specific_mask_directed or args.node_specific_mask_ent_rel or args.node_specific_mask_ent_rel_directed:
                att_mask = [[1]*(2 + 2*len(random_select))]

                for i in range(0, 1+len(random_select)*2):
                    att_mask.append([0]* (2 + 2*len(random_select)))
                att_mask[1][1] = 1              #Head attends to self
            else:
                att_mask = [1]*(2 + 2*len(random_select))
                pad_seq(att_mask, max_seq_len, 0)

            for i, ent_rel in enumerate(random_select):
                path_seq.append(element_vocab[ent_rel[1]])
                path_seq.append(element_vocab[ent_rel[0]])
                pos.append(2)
                pos.append(3)

                if args.node_specific_mask:
                    att_mask[1][2+i*2] = 1      #Head node attends to relation

                    att_mask[2+i*2][1] = 1      #Relation attends to head_node
                    att_mask[2+i*2][2+i*2] = 1  #Relation attends to self
                    att_mask[2+i*2][3+i*2] = 1  #Relation attends to tail

                    att_mask[3+i*2][2+i*2] = 1  #Tail attends to relation
                    att_mask[3+i*2][3+i*2] = 1  #Tail attends to self
                elif args.node_specific_mask_directed:
                    att_mask[1][2+i*2] = 1      #Head node attends to relation

                    att_mask[2+i*2][2+i*2] = 1  #Relation attends to self
                    att_mask[2+i*2][3+i*2] = 1  #Relation attends to tail

                    att_mask[3+i*2][3+i*2] = 1  #Tail attends to self

                elif args.node_specific_mask_ent_rel:
                    att_mask[1][2+i*2] = 1      #Head node attends to relation
                    att_mask[1][3+i*2] = 1      #Head attends to tail

                    att_mask[2+i*2][1] = 1      #Relation attends to head_node
                    att_mask[2+i*2][2+i*2] = 1  #Relation attends to self
                    att_mask[2+i*2][3+i*2] = 1  #Relation attends to tail

                    att_mask[3+i*2][1] = 1      #Tail attends to head
                    att_mask[3+i*2][2+i*2] = 1  #Tail attends to relation
                    att_mask[3+i*2][3+i*2] = 1  #Tail attends to self
                
                elif args.node_specific_mask_ent_rel_directed:
                    att_mask[1][2+i*2] = 1      #Head node attends to relation
                    att_mask[1][3+i*2] = 1      #Head attends to tail

                    att_mask[2+i*2][2+i*2] = 1  #Relation attends to self
                    att_mask[2+i*2][3+i*2] = 1  #Relation attends to tail

                    att_mask[3+i*2][3+i*2] = 1  #Tail attends to self

            if args.node_specific_mask or args.node_specific_mask_directed or args.node_specific_mask_ent_rel or args.node_specific_mask_ent_rel_directed:
                pad_seq(att_mask, max_seq_len, 0, two_dim=True)
                for i in range(0, max_seq_len-len(att_mask)):
                    att_mask.append([0]*max_seq_len)

            for i in range(0, len(random_select)+1):
                attention_masks.append(att_mask)

            pad_seq(path_seq, max_seq_len, element_vocab['[PAD]'])

            orig_inputs.append(path_seq)
            for i in range(0, len(random_select)):
                orig_inputs.append(path_seq)


            pad_seq(pos, max_seq_len, 0)
            positions.append(pos)
            for i in range(0, len(random_select)):
                positions.append(pos)

            center_node_mask = path_seq[:]
            center_node_mask[1] = element_vocab['[MASK]']
            inputs.append(center_node_mask)

            input_labels = [-100] * max_seq_len
            input_labels[1] = path_seq[1]
            labels.append(input_labels)

            for i in range(0, len(random_select)):
                input_labels = [-100] * max_seq_len
                tail_node_mask = path_seq[:]
                tail_node_mask[3+i*2] = element_vocab['[MASK]']
                input_labels[3+i*2] = path_seq[3+i*2]
                labels.append(input_labels)
                inputs.append(tail_node_mask)


def pad_seq(seq, max_seq_len, pad_token, two_dim = False):
    """
    Pads a 1-dim or 2-dim array with a pad_token so shape = (?, max_seq_len)
    """
    if two_dim:
        for sub_seq in seq:
            sub_seq += [pad_token] * (max_seq_len - len(sub_seq))
    else:
         seq += [pad_token] * (max_seq_len - len(seq))


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
