import random
import numpy as np
import torch
import os
import sys
import argparse
import datetime

from datetime import timedelta
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from model import ConKGE
from preprocess_data import preprocess_train_data, preprocess_test_data
from dataset import GraphDataset
from train import train_model
from evaluate import evaluate
from predict import predict

from timeit import default_timer as timer


def run(args):

    print(datetime.datetime.now())
    start = timer()

    print("\n Args: ")
    print(args)

    if torch.cuda.is_available():
        print("Cuda is avaiable... device = cuda")
        device = torch.device("cuda")
    else:
        print("Cuda is not avaiable... device = cpu")
        device = torch.device("cpu")

    set_seed()

    print("Loading the dataset into a graph object...", end=' ')
    ## Initialize dataset
    train_path = os.path.join(args.data_dir, args.train_file)
    test_path = os.path.join(args.data_dir, args.test_file)
    valid_path = os.path.join(args.data_dir, args.valid_file)
    dataset = GraphDataset(args.dataset_name, train_path, test_path, valid_path)
    print("Done")

    ## Preprocess train data
    print("Pre-processing training set...", end=" ")
    train_dataloader = preprocess_train_data(args, dataset.train_graph, dataset.element_vocab, args.max_seq_len, args.batch_size, rand=True)
    print("Done")

    ## Preprocess validation data
    print("Pre-processing validation set...", end=" ")
    valid_dataloader = preprocess_test_data(args, dataset.valid_graph, dataset.element_vocab, args.batch_size)

    print("Done")

    ## Preprocess test data
    print("Pre-processing test set...", end=" ")
    test_dataloader = preprocess_test_data(args, dataset.test_graph, dataset.element_vocab, args.batch_size)

    print("Done")

    end = timer()
    seconds = end-start
    print("{:0>2}".format(str(timedelta(seconds=seconds))))
    

    ## Initialize model
    config = BertConfig()
    
    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.num_attention_heads = args.num_attention_heads
    config.num_hidden_layers = args.num_hidden_layers
    print(config)
    model = ConKGE(config, len(dataset.element_vocab))

    model.to(device)

    print("Model intialized")

    ## Initialize optimizer
    optimizer = AdamW(model.parameters())
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    start_epoch = 0

    if args.checkpoint_file:
        print("Loading checkpoint '{}'...".format(args.checkpoint_file), end=' ')
        checkpoint_path = os.path.join(args.experiment_dir, args.checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps,
                                                last_epoch=start_epoch-1)
        print(" Checkpoint loaded for epoch ", start_epoch)

    # Start training
    train_model(args, model, train_dataloader, valid_dataloader, optimizer, scheduler, device, dataset, start_epoch, args.epochs)

    print("Saving (entire) final model...", end=" ")
    model_path = os.path.join(args.experiment_dir, 'model.pt')
    torch.save(model, model_path)
    print("Done")
    preds, true_inputs, true_labels = predict(args, model, test_dataloader, device)
    print("Evaluate on test set: ")
    mrr, hits_1, hits_3, hits_10, ratio_h1, ratio_h3, ratio_h10 = evaluate(preds, true_inputs, true_labels, dataset)
    print('{{"metric": "Mean reciprocal rank", "value": {}}}'.format(mrr))
    print('{{"metric": "hits_1 (Total)", "value": {}}}'.format(hits_1))
    print('{{"metric": "hits_1 (Ratio)", "value": {}}}'.format(ratio_h1))
    print('{{"metric": "hits_3 (Total)", "value": {}}}'.format(hits_3))
    print('{{"metric": "hits_3 (Ratio)", "value": {}}}'.format(ratio_h3))
    print('{{"metric": "hits_10 (Total)", "value": {}}}'.format(hits_10))
    print('{{"metric": "hits_10 (Ratio)", "value": {}}}'.format(ratio_h10))



def set_seed(seed=42):
    print("Seed set to: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    # Data path, train, test, val name
    # save model - always save to experiment folder?
    # model config
    # model precprocessing settings (Graph choice, attention mask)
    # model evaluation (which evaluations should it do)
    # load_model, model_path, model_name (From checkpoint?)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help="experiment name")
    parser.add_argument('--experiment_dir', type=str, default=None, help='If it exists, the path to the directory where the experiment is contained')
    parser.add_argument('--checkpoint_file', type=str, default=None, help='Name of the checkpoint file to load')

    parser.add_argument('--dataset_name', type=str, default='test', help="Name of the dataset")
    parser.add_argument('--data_dir', type=str, default=None, help="data directory path")
    parser.add_argument('--train_file', type=str, default='train.txt', help="name of file with train data")
    parser.add_argument('--test_file', type=str, default='test.txt', help="name of file with test data")
    parser.add_argument('--valid_file', type=str, default='valid.txt', help="name of file with validation data")


    parser.add_argument('--hidden_size', type=int, default=256, help="Model config: hidden size, default=256")
    parser.add_argument('--intermediate_size', type=int, default=512, help="Model config: intermediate size (Dim of FFNN in transformer layer), default=512")
    parser.add_argument('--num_attention_heads', type=int, default=4, help="Model config: number of attention heads, default=4")
    parser.add_argument('--num_hidden_layers', type=int, default=6, help="Model config: number of transformer layers, default=4")

    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for AdamW optimizer")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--warmup_proportion', type=float, default=0.0, help="Proportion of training steps for warmup.")

    parser.add_argument('--max_seq_len', type=int, default=128, help="The max length for a sequence")
    parser.add_argument('--batch_size', type=int, default=512, help="The size of a single batch")
    parser.add_argument('--node_specific_mask', type=bool, default=False, help="If set true, a node specific attention mask is applied where each node only attend to its neighborhood")
    parser.add_argument('--node_specific_mask_directed', type=bool, default=False, help="If set true, a node specific attention mask is applied where each node only attend to its neighborhood in the direction of the graph")
    parser.add_argument('--node_specific_mask_ent_rel', type=bool, default=False, help="If set true, a node specific attention mask is applied where each node only attend to its 2-hop neighborhood adjacent entities and rels")
    parser.add_argument('--node_specific_mask_ent_rel_directed', type=bool, default=False, help="If set true, a node specific attention mask is applied where each node only attend to its 2-hop neighborhood adjacent entities and rels in the direction of the graph")


    parser.add_argument('--triples_head_tail', type=bool, default=False, help='Adds all triples twice to the dataset, with the entity head/tail node masked')
    parser.add_argument('--triples_rel', type=bool, default=False, help='Adds all triples to the dataset with the rel node masked')

    parser.add_argument('--path_length', type=int, default=0, help='Indicates if paths should be added to training data and the length of paths. Length=1 is the extenstion of a triple.')
    parser.add_argument('--subgraph_triples', type=bool, default=False, help='Creates a graph of connected triples.')

    parser.add_argument('--num_rand_paths', type=int, default=1, help='Indicates how many random paths should be created from a triple.')
    parser.add_argument('--num_samples', type=int, default=1, help='Indicates how many graphs of connected triples should be created from each node.')


    return parser.parse_args()

def setup_experiment(args):

    # Create directory for experiment
    path_dir = os.path.join('experiments', args.name)
    new_path_dir = path_dir
    count = 0
    while os.path.exists(new_path_dir):
        count += 1
        new_path_dir = path_dir+'-'+str(count)
    
    os.mkdir(new_path_dir)
    args.experiment_dir = new_path_dir

    return new_path_dir



if __name__ == '__main__':
    args = parse_args()

    if args.experiment_dir == None:
        setup_experiment(args)

    run(args)
