import json
import pickle
import time
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import nltk
import config

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


class SQuadDataset(data.Dataset):
    def __init__(self, src_file, trg_file, max_length, word2idx, debug=False):
        self.src = open(src_file, "r").readlines()
        self.trg = open(trg_file, "r").readlines()

        assert len(self.src) == len(self.trg), \
            "the number of source sequence {}" " and target sequence {} must be the same" \
            .format(len(self.src), len(self.trg))

        self.max_length = max_length
        self.word2idx = word2idx
        self.num_seqs = len(self.src)

        if debug:
            self.src = self.src[:100]
            self.trg = self.trg[:100]
            self.num_seqs = 100

    def __getitem__(self, index):
        src_seq = self.src[index]
        trg_seq = self.trg[index]
        src_seq, ext_src_seq, oov_lst = self.context2ids(
            src_seq, self.word2idx)
        trg_seq, ext_trg_seq = self.question2ids(
            trg_seq, self.word2idx, oov_lst)
        return src_seq, ext_src_seq, trg_seq, ext_trg_seq, oov_lst

    def __len__(self):
        return self.num_seqs

    def context2ids(self, sequence, word2idx):
        ids = list()
        extended_ids = list()
        oov_lst = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                extended_ids.append(len(word2idx) + oov_lst.index(token))
        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.tensor(ids, dtype=torch.long)
        extended_ids = torch.tensor(extended_ids, dtype=torch.long)

        return ids, extended_ids, oov_lst

    def question2ids(self, sequence, word2idx, oov_lst):
        ids = list()
        extended_ids = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token in oov_lst:
                    extended_ids.append(len(word2idx) + oov_lst.index(token))
                else:
                    extended_ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)
        return ids, extended_ids

def collate_fn(data):
    def merge(sequences):
        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst = zip(*data)

    src_seqs = merge(src_seqs)
    ext_src_seqs = merge(ext_src_seqs)
    trg_seqs = merge(trg_seqs)
    ext_trg_seqs = merge(ext_trg_seqs)
    return src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst

class SQuadDatasetWithTag(data.Dataset):
    def __init__(self, src_file, trg_file, bio_file, adj_file, ner_file, an_file, pos_file, max_length, word2idx, debug=False):
        self.srcs = []
        self.tags = []
        self.ners = []
        self.styles = []
        self.ans = []
        self.poses = []
        self.ans = open(an_file, "r").readlines()

        sentence = []
        with open(src_file,'r') as lines:
            for line in lines:
                line = line.strip()
                tokens = line.split(" ")
                tokens.insert(0, START_TOKEN)
                tokens.append(END_TOKEN)
                sentence.append(tokens)

        self.srcs = sentence

        lines = open(bio_file, "r").readlines()
        # tags = []
        self.entity2idx = {"O": 0, "B": 1, "I": 2}
        for line in lines:
            tags = []

            line = line.strip()
            line = line.split(" ")

            tags.insert(0, self.entity2idx["O"])
            # tokens.append(tokens)
            for bio in line:
                if bio == "0":
                    bio = "O"

                tags.append(self.entity2idx[bio])
            tags.append(self.entity2idx["O"])
            self.tags.append(tags)


        lines = open(ner_file, "r").readlines()
        # tags = []
        self.entity2idx = {"o": 0, "misc": 1, "number": 2, "location":3, "person": 4, "ordinal": 5, "duration": 6,\
                           "time": 7, "date": 8, "organization": 9, "percent": 10, "money": 11
                           }
        for line in lines:
            ner = []

            line = line.strip()
            line = line.split(" ")
            ner.insert(0, self.entity2idx["o"])
            for n in line:
                ner.append(self.entity2idx[n])
            ner.append(self.entity2idx["o"])

            self.ners.append(ner)

        lines = open(pos_file, "r").readlines()
        # tags = []
        self.entity2idx = {'<s>': 0, 'NNP': 1, 'CD': 2, 'MD': 3, 'VB': 4, 'DT': 5,
                           'NN': 6, 'IN': 7, ',': 8, 'JJ': 9, 'CC': 10,
                           'NNS': 11, '.': 12, 'PRP': 13, 'VBZ': 14, 'VBN': 15,
                           'VBD': 16, 'TO': 17, ':': 18, 'RP': 19, 'RB': 20,
                           'VBG': 21, 'PRP$': 22, 'NNPS': 23, 'POS': 24, '``': 25,
                           "''": 26, 'WDT': 27, 'WP': 28, 'JJS': 29, 'JJR': 30,
                           'VBP': 31, 'EX': 32, '-LRB-': 33, '-RRB-': 34, 'RBR': 35,
                           'FW': 36, 'RBS': 37, 'WP$': 38, 'WRB': 39, '$': 40, 'PDT': 41,
                           '#': 42, 'UH': 43, 'LS': 44, 'SYM': 45, '</s>': 46}
        for line in lines:
            pos = []
            # print("line:",line)
            line = line.strip()
            line = line.split(" ")

            pos.insert(0, self.entity2idx["<s>"])

            for po in line:

                pos.append(self.entity2idx[po])
            pos.append(self.entity2idx["</s>"])
            self.poses.append(pos)

        self.trgs = open(trg_file, "r").readlines()

        with open(adj_file, "r") as f:
            load_dict = json.load(f)
            self.adj = load_dict["adj"]


        assert len(self.srcs) == len(self.trgs), \
            "the number of source sequence {}" " and target sequence {} must be the same" \
            .format(len(self.srcs), len(self.trgs))

        self.max_length = max_length
        self.word2idx = word2idx
        self.num_seqs = len(self.srcs)

        if debug:
            self.srcs = self.srcs[:100]
            self.trgs = self.trgs[:100]
            self.tags = self.tags[:100]
            self.ners = self.ners[:100]
            self.styles = self.styles[:100]
            self.ans = self.ans[:100]
            self.poses = self.poses[:100]
            self.num_seqs = 100

    def __getitem__(self, index):
        src_seq = self.srcs[index]
        trg_seq = self.trgs[index]
        tag_seq = self.tags[index]
        adj_seq = self.adj[index]
        ner_seq = self.ners[index]

        an_seq = self.ans[index]
        pos_seq = self.poses[index]

        tag_seq = torch.Tensor(tag_seq[:self.max_length])
        ner_seq = torch.Tensor(ner_seq[:self.max_length])
        pos_seq = torch.Tensor(pos_seq[:self.max_length])
        src_seq, ext_src_seq, oov_lst = self.context2ids(
            src_seq, self.word2idx)
        trg_seq, ext_trg_seq = self.question2ids(
            trg_seq, self.word2idx, oov_lst)

        an_seq = self.an_2ids(
            an_seq, self.word2idx, oov_lst)
        adj_seq = torch.Tensor(adj_seq)

        return src_seq, ext_src_seq, trg_seq, ext_trg_seq, oov_lst, tag_seq, adj_seq, ner_seq, an_seq, pos_seq

    def __len__(self):
        return self.num_seqs

    def context2ids(self, tokens, word2idx):
        ids = list()
        extended_ids = list()
        oov_lst = list()

        # START and END token is already in tokens lst
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            if len(ids) == self.max_length:
                break

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids, oov_lst

    def question2ids(self, sequence, word2idx, oov_lst):
        ids = list()
        extended_ids = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token in oov_lst:
                    extended_ids.append(len(word2idx) + oov_lst.index(token))
                else:
                    extended_ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids

    def question_style_2ids(self, sequence, word2idx, oov_lst):
        ids = list()
        ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)

        return ids

    def an_2ids(self, sequence, word2idx, oov_lst):
        ids = list()
        ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)

        return ids

def collate_fn_emotion(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = 1

    batch_seq = np.array([inst.numpy() for inst in insts])

    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq,

def collate_fn_tag(data):
    def merge(sequences):

        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs

# ================src_seq = 55 ===================
    def merge55(sequences):
        lengths = [len(sequence) for sequence in sequences]
        src_len = 55

        padded_seqs = torch.zeros(len(sequences), src_len).long()

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs
# ====================================================
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst, tag_seqs, adj_seq, ner_seqs, an_seq, pos_seq = zip(
        *data)
    an_seq = merge(an_seq)
    adj_seq = collate_fn_emotion(adj_seq)
    src_seqs = merge55(src_seqs)
    ext_src_seqs = merge55(ext_src_seqs)
    trg_seqs = merge(trg_seqs)

    ext_trg_seqs = merge(ext_trg_seqs)
    tag_seqs = merge55(tag_seqs)
    ner_seqs = merge55(ner_seqs)
    pos_seq = merge55(pos_seq)

    assert src_seqs.size(1) == tag_seqs.size(
        1), "length of tokens and tags should be equal"


    return src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, tag_seqs, oov_lst, adj_seq, ner_seqs, an_seq, pos_seq


def get_loader(src_file, trg_file, bio_file, adj_file, ner_file, an_file, pos_file, word2idx,
               batch_size, use_tag=False, debug=False, shuffle=False):
    dataset = SQuadDatasetWithTag(src_file, trg_file, bio_file, adj_file, ner_file, an_file, pos_file, config.max_len,
                                  word2idx, debug)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn_tag)

    return dataloader


def make_vocab(src_file, trg_file, dev_src_file, dev_trg_file, output_file, max_vocab_size):
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3
    counter = dict()
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1
    with open(trg_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1
    with open(dev_src_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1
    with open(dev_trg_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    for i, (word, _) in enumerate(sorted_vocab, start=4):
        if i == max_vocab_size:
            break
        word2idx[word] = i
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def make_vocab_from_squad(output_file, counter, max_vocab_size):
    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3

    for idx, (token, freq) in enumerate(sorted_vocab, start=4):
        if len(word2idx) == max_vocab_size:
            break
        word2idx[token] = idx
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def make_embedding(embedding_file, output_file, word2idx):
    word2embedding = dict()
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    num_oov = 0
    for word, idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
    return embedding


def time_since(t):
    """ Function for time. """
    return time.time() - t


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)


def outputids2words(id_list, idx2word, article_oovs=None):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param article_oovs: list of oov words
    :return: list of words
    """
    words = []
    for idx in id_list:
        try:
            word = idx2word[idx]
        except KeyError:
            if article_oovs is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = article_oovs[article_oov_idx]
                except IndexError:
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ID]
        words.append(word)

    return words


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
