import sys

sys.path.insert(0, '../')

import config
from data_utils import (make_embedding, make_vocab)




def make_sent_dataset():

    train_src_file = "../squad/train_src50.txt"
    train_trg_file = "../squad/train_tgt50.txt"
    # dev file
    dev_src_file = "../squad/dev_src50.txt"
    dev_trg_file = "../squad/dev_tgt50.txt"

    embedding_file = "./glove.840B.300d.txt"
    embedding = "./embedding.pkl"
    word2idx_file = "./word2idx.pkl"
    # make vocab file
    word2idx = make_vocab(train_src_file, train_trg_file, dev_src_file, dev_trg_file, word2idx_file, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)



if __name__ == "__main__":
    make_sent_dataset()
