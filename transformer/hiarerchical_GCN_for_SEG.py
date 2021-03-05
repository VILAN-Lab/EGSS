import torch
import config
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F


class GCN_Module(nn.Module):
    def __init__(self, embeddings, vocab, embedding_size, d_model, hidden_size, d_inner):
        super(GCN_Module, self).__init__()
        self.embedding = nn.Embedding(vocab, embedding_size)
        self.tag_embedding = nn.Embedding(100, 100)
        self.ner_embedding = nn.Embedding(50, 50)
        self.pos_embedding = nn.Embedding(60, 60)
        self.layer6 = nn.Linear(510, 300)
        self.layer7 = nn.Linear(300, 600)
        self.node_number = 55
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)
        # self.node_number_2 = args.node_number_2
        self.intra_sent_gcn1 = Intra_sentence_GCN(hidden_size, 4, self.node_number)
        self.sent1_level_feat = AttFlatten(hidden_size, d_inner)

    def forward(self, sent1, adj1, tag_seq, ner_seq, pos_seq):

        batch_size, n_words = sent1.size()
        sent1 = sent1.to(config.device)

        sent1 = self.embedding(sent1)[:, :n_words]
        tag_seq = self.tag_embedding(tag_seq)[:, :n_words]
        ner_seq = self.ner_embedding(ner_seq)[:, :n_words]
        pos_seq = self.pos_embedding(pos_seq)[:, :n_words]
        sent1_tag_emb = torch.cat((sent1, tag_seq), 2)
        sent1_tag_ner_emb = torch.cat((sent1_tag_emb, ner_seq), 2)
        sent1_tag_ner_pos_emb = torch.cat((sent1_tag_ner_emb, pos_seq), 2)
        sent1 = self.layer6(sent1_tag_ner_pos_emb)

        sent1_feat = self.intra_sent_gcn1(sent1, adj1)  # (batch, n_word, hidden_size)

        encoder_output = sent1_feat + sent1

        return encoder_output

    def mask_for_sentence(self, sentence, sent):

        return (torch.sum(torch.abs(sentence), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class Intra_sentence_GCN(nn.Module):
    def __init__(self, mem_dim, layers, number_node):
        super(Intra_sentence_GCN, self).__init__()

        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(0.1)

        # linear transformation
        self.linear_output = weight_norm(nn.Linear(self.mem_dim, self.mem_dim), dim=None)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(weight_norm(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim), dim=None))

        self.weight_list = self.weight_list.to(config.device)
        self.linear_output = self.linear_output.to(config.device)

        self.linear_node = nn.Linear(number_node, number_node).to(config.device)
        self.relu = nn.ReLU()

    def forward(self, gcn_inputs, adj):
        # gcn layer
        denom = (adj.sum(2).unsqueeze(2)).float()

        outputs = gcn_inputs.float()

        cache_list = [outputs]
        output_list = []
        adj = adj.float().to(config.device)

        for i in range(self.layers):

            # relative_score = outputs.bmm(outputs.transpose(1, 2))
            # relative_score = self.linear_node(relative_score).to(config.device)
            # relative_score = torch.softmax(relative_score, dim=-1)

            # adj = adj.bmm(relative_score)
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[i](Ax)
            AxW = AxW + self.weight_list[i](outputs)  # self loop
            AxW = AxW / denom
            gAxW = self.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)

        return out

class Inter_sentence_GCN(nn.Module):
    def __init__(self, mem_dim, layers, number_node):
        super(Inter_sentence_GCN, self).__init__()

        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(0.1)

        # linear transformation
        self.linear_output = weight_norm(nn.Linear(self.mem_dim, self.mem_dim), dim=None)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(weight_norm(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim), dim=None))

        self.weight_list = self.weight_list.to(config.device)
        self.linear_output = self.linear_output.to(config.device)

        self.linear_node = nn.Linear(number_node, number_node).to(config.device)
        self.relu = nn.ReLU()

    def forward(self, gcn_inputs, adj):
        # gcn layer
        denom = (adj.sum(2).unsqueeze(2)).float()
        outputs = gcn_inputs.float()
        cache_list = [outputs]
        output_list = []
        adj = adj.float().to(config.device)

        for i in range(self.layers):

            relative_score = outputs.bmm(outputs.transpose(1, 2))
            relative_score = self.linear_node(relative_score).to(config.device)
            relative_score = torch.softmax(relative_score, dim=-1)

            adj = adj.bmm(relative_score)
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[i](Ax)
            AxW = AxW + self.weight_list[i](outputs)  # self loop
            AxW = AxW / denom
            gAxW = self.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out


class AttFlatten(nn.Module):
    def __init__(self, hidden_size, d_inner):
        super(AttFlatten, self).__init__()
        self.hidden_size = hidden_size
        self.d_inner = d_inner

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=d_inner,
            out_size=1,
            dropout_r=0.1,
            use_relu=True
        )

        self.linear_merge = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)

        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FullyConnectedLayer(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FullyConnectedLayer, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x