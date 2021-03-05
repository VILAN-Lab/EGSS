import argparse
import torch
import transformer.Constants as Constants
import json
from relation_to_adj_matrix import tree_to_adj, head_to_tree
from tqdm import tqdm
import numpy as np
import jsonlines

def relation_to_adj_matrix(relation, name, sent):

    if name == 'train':

        head = head_to_tree(relation)

        if sent == 'sent1':
            adj_mat = tree_to_adj(55, head, sent)


    elif name == 'val':

        head = head_to_tree(relation)

        if sent == 'sent1':
            adj_mat = tree_to_adj(55, head, sent)

    else:

        head = head_to_tree(relation)

        if sent == 'sent1':
            adj_mat = tree_to_adj(55, head, sent)

    return adj_mat

train_row_relation = './squad/dev-split1-8.25-treedependency50.json'

#====================start============================
with jsonlines.open(train_row_relation, mode='r') as reader:
    adj1 = []
    for s in reader:

        a = (relation_to_adj_matrix(s['sent1'][0], "train", 'sent1')).tolist()
        adj1.append(a)

result ={'adj': adj1}


with open("./squad/dev_split1.1_adj50.txt", "w") as fp:
    fp.write(json.dumps(result, indent=4))

