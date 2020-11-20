import random
import numpy as np
import torch
import os

from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from model import ConKGE
from preprocess_data import preprocess_data
from dataset import GraphDataset
from train import train_model
from evaluate import predict, hits

from timeit import default_timer as timer


BATCH_SIZE = 2048

def run():

    if torch.cuda.is_available():
        print("Cuda is avaiable... device = cuda")
        device = torch.device("cuda")
    else:
        print("Cuda is not avaiable... device = cpu")
        device = torch.device("cpu")


    set_seed()
    print("Seed set")

    ## Initialize dataset
    DIR_PATH = 'data/fb15k/'
    TRAIN_PATH = DIR_PATH + 'train.txt'
    TEST_PATH = DIR_PATH + 'test.txt'
    VALID_PATH = DIR_PATH + 'valid.txt'
    dataset = GraphDataset("fb15k", TRAIN_PATH, TEST_PATH, VALID_PATH)

    print("Dataset initialized")

    ## Preprocess train data
    train_dataloader = preprocess_data(dataset.train_graph, dataset.ent_vocab, dataset.rel_vocab, BATCH_SIZE, rand=True)

    ## Preprocess validation data
    valid_dataloader = preprocess_data(dataset.valid_graph, dataset.ent_vocab, dataset.rel_vocab, BATCH_SIZE, eval_set=True)

    ## Preprocess test data
    test_dataloader = preprocess_data(dataset.test_graph, dataset.ent_vocab, dataset.rel_vocab, BATCH_SIZE, eval_set=True)

    print("Data pre-processed")

    ## Initialize model
    config = BertConfig()
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.num_hidden_layers = 6
    model = ConKGE(config, len(dataset.ent_vocab), len(dataset.rel_vocab))
    model.to(device)

    print("Model intialized")

    ## Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    NO_EPOCHS = 100
    total_steps = len(train_dataloader) * NO_EPOCHS
   
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    # Start training
    train_model(model, train_dataloader, valid_dataloader, optimizer, scheduler, device, dataset, NO_EPOCHS)
    
    preds, true_inputs, true_labels = predict(model, test_dataloader, device)
    
    print("Evaluate on test set: ")
    hits_1, hits_3, hits_10, total, ratio_h1, ratio_h3, ratio_h10 = hits(preds, true_inputs, true_labels, dataset)
    print("TOTAL: ", total)
    print("HITS@1: ", hits_1, ratio_h1)
    print("HITS@3: ", hits_3, ratio_h3)
    print("HITS@10", hits_10, ratio_h10)



def set_seed(seed=42):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



if __name__ == '__main__':
    run()