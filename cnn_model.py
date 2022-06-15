import torch
import numpy as np
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *

class CNNModel(nn.Module):
    def __init__(self, label_size, vocab, words_num, device, keep_dropout=0.2, pretrained_emb=None, freeze_emb=True, vocab_size=50000):
        super(CNNModel, self).__init__()

        self.keep_dropout = keep_dropout
        self.in_channel = 1
        self.kernel_nums = 128
        self.kernel_sizes = (3, 4, 5)
        self.vocab_size = vocab_size
        self.embedding_dim = 300
        self.label_size = label_size
        self.vocab = vocab
        self.words_num = words_num
        self.device = device

        if pretrained_emb is not None:
            self.embs = nn.Embedding.from_pretrained(torch.tensor(pretrained_emb, dtype=torch.float32), freeze=freeze_emb)
        else:
            self.embs = nn.Embedding(self.vocab_size+1, self.embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(self.in_channel, self.kernel_nums, (K, self.embedding_dim)) for K in self.kernel_sizes])

        self.dropout = nn.Dropout(self.keep_dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.kernel_nums, self.label_size)

    def forward(self, x):
        """ Just used for virtual adversarial attacks!!! """

        # Conv & max pool
        # x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = self.fc(x)  # (batch_size, num_aspects)

        return logit

    def init_weight(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=0.1)
            torch.nn.init.constant_(conv.bias, 0.1)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)

    def input_to_embs(self, x):
        # Embedding
        x = self.embs(x) # dim: (batch_size, max_seq_len, embedding_size)
        return x

    def embs_to_logit(self, x, mask=None):

        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = self.fc(x)  # (batch_size, num_aspects)

        return logit

    def input_to_logit(self, x):
        # Embedding
        x = self.embs(x) # dim: (batch_size, max_seq_len, embedding_size)

        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = self.fc(x)  # (batch_size, num_aspects)

        return logit

    def query(self, sentences, labels):
        x, _ = text_encoder(sentences, self.vocab, self.words_num)
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        logits = self.input_to_logit(x).detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        return logits, predictions

    def query_from_ids(self, x):
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        logits = self.input_to_logit(x).detach().cpu().numpy()
        return logits

    def get_embeddings(self):
        return self.embs.weight

    
class BiLSTMModel(nn.Module):
    def __init__(self, label_size, vocab, words_num, device, keep_dropout=0.2, pretrained_emb=None, freeze_emb=True, vocab_size=50000):
        super(BiLSTMModel, self).__init__()

        self.keep_dropout = keep_dropout
        self.cell_num = 128
        self.vocab_size = vocab_size
        self.embedding_dim = 300
        self.label_size = label_size
        self.vocab = vocab
        self.words_num = words_num
        self.device = device
        self.bilstm_layer = 1

        if pretrained_emb is not None:
            self.embs = nn.Embedding.from_pretrained(torch.tensor(pretrained_emb, dtype=torch.float32), freeze=freeze_emb)
        else:
            self.embs = nn.Embedding(self.vocab_size+1, self.embedding_dim)

        self.bilstm = nn.LSTM(self.embedding_dim, self.cell_num, num_layers=self.bilstm_layer, bidirectional=True)

        self.fc = nn.Linear(2 * self.cell_num, self.label_size)
        # self.dropout = nn.Dropout(self.keep_dropout)

        self.hidden = None

        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)

    def init_hidden(self, batch_size=None):
        
        h0 = torch.zeros(2*self.bilstm_layer, batch_size, self.cell_num, device=self.device)
        c0 = torch.zeros(2*self.bilstm_layer, batch_size, self.cell_num, device=self.device)

        return (h0, c0)

    def input_to_embs(self, x):
        # Embedding
        x = self.embs(x) # dim: (batch_size, max_seq_len, embedding_size)
        return x

    def embs_to_logit(self, x, mask=None):

        # Conv & max pool
        x = x.permute(1, 0, 2)  # dim: (batch_size, embedding_size, max_seq_len)

        self.hidden = self.init_hidden(x.size()[1]) # dim: (2, batch_size, cell_num)
        # lstm_out, self.hidden = self.bilstm(x, self.hidden) # lstm_out: (max_seq_len, batch_size, cell_num*2)
        lstm_out, self.hidden = self.bilstm(x)
        out = lstm_out.permute(1, 0, 2) # (batch_size, max_seq_len, cell_num * 2)
        final = torch.mean(out, 1)
        # final = self.dropout(final)
        logit = self.fc(final) # batch_size, label_size

        return logit

    def input_to_logit(self, x):
        # Embedding
        x = self.embs(x) # dim: (batch_size, max_seq_len, embedding_size)
        logit = self.embs_to_logit(x)

        return logit

    def query(self, sentences, labels):
        x, _ = text_encoder(sentences, self.vocab, self.words_num)
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        logits = self.input_to_logit(x).detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        return logits, predictions

    def query_from_ids(self, x):
        x = torch.tensor(x, dtype=torch.long).to(self.device)
        logits = self.input_to_logit(x).detach().cpu().numpy()
        return logits

    def get_embeddings(self):
        return self.embs.weight
