#!/bin/bash

# Following is final experiments, all with default settings for AdamW

# Full attention
## Triples
echo "Triples on fb15k"
python3 run.py --name fb15k_triples_20epochs_fullatt --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048
echo "Paths on fb15k"
python3 run.py --name fb15k_paths_20epochs_fullatt --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048
echo "Connected triples on fb15k"
python3 run.py --name fb15k_contrip_20epochs_fullatt --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --num_samples 5

# Node specific attention tests (Each for 20 epochs)
## Node specific attention:
### Triples
echo "Triples bi-directional"
python3 run.py --name fb15k_triples_20epochs_nsatt --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048 --node_specific_mask True
#### Paths
echo "Paths bi-directional"
python3 run.py --name fb15k_paths_20epochs_nsatt --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --node_specific_mask True
### Connected triples
echo "Connected triples bi-directional"
python3 run.py --name fb15k_contrip_20epochs_nsatt --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --node_specific_mask True --num_samples 5

## Directed node specific attention:
### Triples
echo "Triples directed"
python3 run.py --name fb15k_triples_20epochs_nsatt_directed --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048 --node_specific_mask_directed True
#### Paths
echo "Paths directed"
python3 run.py --name fb15k_paths_20epochs_nsatt_directed --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --node_specific_mask_directed True
### Connected triples
echo "Connected triples directed"
python3 run.py --name fb15k_contrip_20epochs_nsatt_directed --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --node_specific_mask_directed True --num_samples 5

## Entity and relation node specific attention:
### Triples
echo "Triples entrel"
python3 run.py --name fb15k_triples_20epochs_nsatt_ent_rel --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048 --node_specific_mask_ent_rel True
### Paths
echo "Paths ent rel"
python3 run.py --name fb15k_paths_20epochs_nsatt_ent_rel --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --node_specific_mask_ent_rel True
### Connected triples
echo "Con triples ent rel"
python3 run.py --name fb15k_contrip_20epochs_nsatt_ent_rel --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --node_specific_mask_ent_rel True --num_samples 5

## Directed entity and relation node specific attention:
### Triples
echo "Triples directed ent rel"
python3 run.py --name fb15k_triples_20epochs_nsatt_ent_rel_directed --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048 --node_specific_mask_ent_rel_directed True
#### Paths
echo "Paths directed ent rel"
python3 run.py --name fb15k_paths_20epochs_nsatt_ent_rel_directed --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --node_specific_mask_ent_rel_directed True
### Connected triples
echo "Connected triples directed ent rel"
python3 run.py --name fb15k_contrip_20epochs_nsatt_ent_rel_directed --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --node_specific_mask_ent_rel_directed True --num_samples 5



# Path variations
## Longer paths
### Path = triple+1
echo "Paths = triple+1"
python3 run.py --name fb15k_paths_20epochs_pl1 --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048
### Path = triple+2
echo "Paths = triple+2"
python3 run.py --name fb15k_paths_20epochs_pl2 --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 2 --max_seq_len 8 --epochs 20 --batch_size 2048
### Path = triple+3
echo "Paths = triple+3"
python3 run.py --name fb15k_paths_20epochs_pl3 --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 3 --max_seq_len 10 --epochs 20 --batch_size 2048
### Path = triple+4
echo "Paths = triple+4"
python3 run.py --name fb15k_paths_20epochs_pl4 --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 4 --max_seq_len 12 --epochs 20 --batch_size 2048

## More paths
### 2 paths
echo "Num random paths = 2"
python3 run.py --name fb15k_paths_20epochs_2randpaths --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --num_rand_paths 2
### 4 paths
echo "Num random paths = 4"
python3 run.py --name fb15k_paths_20epochs_4randpaths --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --num_rand_paths 4
### 6 paths
echo "Num random paths = 6"
python3 run.py --name fb15k_paths_20epochs_6randpaths --dataset_name fb15k --data_dir data/fb15k/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048 --num_rand_paths 6


# Connected triples variants (20 epochs)
#Samples per node = 1
echo "Connected triples, num triples=2"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 6 --num_rand_paths 2 --epochs 20 --batch_size 2048 --node_specific_mask True
echo "Connected triples, num triples=4"
python3 run.py --name fb15k_contrip_20epochs_4paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --node_specific_mask True
echo "Connected triples, num triples=6"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 14 --num_rand_paths 6 --epochs 20 --batch_size 2048

#Samples per node = 3
echo "Connected triples, num triples=2"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 6 --num_rand_paths 2 --epochs 20 --batch_size 2048 --num_samples 3 --node_specific_mask True
echo "Connected triples, num triples=4"
python3 run.py --name fb15k_contrip_20epochs_4paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --num_samples 3 --node_specific_mask True
echo "Connected triples, num triples=6"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 14 --num_rand_paths 6 --epochs 20 --batch_size 2048 --num_samples 3

#Samples per node = 5
echo "Connected triples, num triples=2"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 6 --num_rand_paths 2 --epochs 20 --batch_size 2048 --num_samples 5 --node_specific_mask True
echo "Connected triples, num triples=4"
python3 run.py --name fb15k_contrip_20epochs_4paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --num_samples 5 --node_specific_mask True
echo "Connected triples, num triples=6"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 14 --num_rand_paths 6 --epochs 20 --batch_size 2048 --num_samples 5

#Samples per node = 7
echo "Connected triples, num triples=2"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 6 --num_rand_paths 2 --epochs 20 --batch_size 2048 --num_samples 7 --node_specific_mask True
echo "Connected triples, num triples=4"
python3 run.py --name fb15k_contrip_20epochs_4paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --num_samples 7 --node_specific_mask True
echo "Connected triples, num triples=6"
python3 run.py --name fb15k_contrip_20epochs_2paths --dataset_name fb15k --data_dir data/fb15k/ --subgraph_triples True --max_seq_len 14 --num_rand_paths 6 --epochs 20 --batch_size 2048 --num_samples 7



# Diff data (20 epochs)
echo "Triples on wn18"
python3 run.py --name wn18_triples_20epochs_fullatt --dataset_name wn18 --data_dir data/wn18/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048
echo "Triples on fb15k237"
python3 run.py --name fb15k237_triples_20epochs_fullatt --dataset_name fb15k237 --data_dir data/fb15k237/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048
echo "Triples on wn18rr"
python3 run.py --name wn18rr_triples_20epochs_fullatt --dataset_name wn18rr --data_dir data/wn18rr/ --triples_head_tail True --max_seq_len 4 --epochs 20 --batch_size 2048 
## Paths

echo "Paths on wn18"
python3 run.py --name wn18_paths_20epochs_fullatt --dataset_name wn18 --data_dir data/wn18/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048
echo "Paths on fb15k237"
python3 run.py --name fb15k237_paths_20epochs_fullatt --dataset_name fb15k237 --data_dir data/fb15k237/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048
echo "Paths on wn18rr"
python3 run.py --name wn18rr_paths_20epochs_fullatt --dataset_name wn18rr --data_dir data/wn18rr/ --triples_head_tail True --path_length 1 --max_seq_len 6 --epochs 20 --batch_size 2048
## Connected triples

echo "Connected triples on wn18"
python3 run.py --name wn18_contrip_20epochs_fullatt --dataset_name wn18 --data_dir data/wn18/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --num_samples 5
echo "Connected triples on fb15k237"
python3 run.py --name fb15k237_contrip_20epochs_fullatt --dataset_name fb15k237 --data_dir data/fb15k237/ --subgraph_triples True --max_seq_len 10 --num_rand_paths 4 --epochs 20 --batch_size 2048 --num_samples 5
echo "Connected triples on wn18rr"
python3 run.py --name wn18rr_contrip_20epochs_fullatt --dataset_name wn18rr --data_dir data/wn18rr/ --subgraph_triples True --max_seq_len 8 --num_rand_paths 3 --epochs 20 --batch_size 2048 --num_samples 5
