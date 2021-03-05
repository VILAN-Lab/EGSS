
# train file
train_src_file = "./squad/train_src50.txt"
train_trg_file = "./squad/train_tgt50.txt"
train_bio_file = "./squad/train_bio50.txt"
train_adj_file = "./squad/train_adj.txt"
train_ner_file = "./squad/train_ner50.txt"
train_an_file = "./squad/train_other-ans-style.txt"
train_pos_file = "./squad/train_pos50.txt"
# dev file
dev_src_file = "./squad/dev_src50.txt"
dev_trg_file = "./squad/dev_tgt50.txt"
dev_bio_file = "./squad/dev_bio50.txt"
dev_adj_file = "./squad/dev_adj.txt"
dev_ner_file = "./squad/dev_ner50.txt"
dev_an_file = "./squad/dev_other-ans-style.txt"
dev_pos_file = "./squad/dev_pos50.txt"
# test file
test_src_file = "./squad/test_src50.txt"
test_trg_file = "./squad/test_tgt50.txt"
test_bio_file = "./squad/test_bio50.txt"
test_adj_file = "./squad/test_adj.txt"
test_ner_file = "./squad/test_ner50.txt"
test_an_file = "./squad/test_other-ans-style.txt"
test_pos_file = "./squad/test_pos50.txt"

embedding = "./data/embedding.pkl"
word2idx_file = "./data/word2idx.pkl"

model_path = "./save/model.pt"
train = True
device = "cuda:0"
use_gpu = True
debug = False
vocab_size = 45000
freeze_embedding = True

num_epochs = 30
max_len = 100
num_layers = 2
hidden_size = 300
d_inner = 2048
embedding_size = 300
d_model = 600
lr = 0.1
batch_size = 64
dropout = 0.2
max_grad_norm = 5.0

use_pointer = True
beam_size = 12
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/"
