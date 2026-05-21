
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

def time_embedding(t, hidden_dim, seq_len, device):
    t = t.view(-1, 1).float()             
    te = torch.zeros(t.size(0), hidden_dim, device=device)
    div_term = 1 / (10000 ** (torch.arange(0, hidden_dim, 2, device=device) / hidden_dim))
    te[:, 0::2] = torch.sin(t * div_term)
    te[:, 1::2] = torch.cos(t * div_term)
    return te.unsqueeze(1).repeat(1, seq_len, 1)

class Attention(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_dim = opt.input_dim
        self.cond_dim = opt.cond_dim
        self.hidden_dim = opt.hidden_dim
        self.nhead = opt.nhead
        self.seq_len = opt.seq_len
        self.device = opt.device

        self.cond_embedder = nn.Sequential(nn.Linear(self.cond_dim, self.hidden_dim),nn.Tanh())
        self.input_projector = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=1,batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,nhead=self.nhead,dim_feedforward=self.hidden_dim) # the number of heads
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=opt.depth ) # add depth
        self.output_projector = nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.output_projector.weight)

    def forward(self, x, c, t):
        time_emb = time_embedding(t, self.hidden_dim, self.seq_len, self.device)
        cond_emb = self.cond_embedder(c)                   
        cond_emb = cond_emb.unsqueeze(1).repeat(1, self.seq_len, 1)  
        hid_enc, _ = self.input_projector(x)       
        h = hid_enc + time_emb + cond_emb
        h2 = self.trans_encoder(h)                  
        output = self.output_projector(h2.permute(0,2,1))  
        return output.permute(0,2,1)            
    
class CNN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_dim  = opt.input_dim
        self.hidden_dim = opt.hidden_dim
        self.seq_len    = opt.seq_len
        self.device     = opt.device

        self.input_projector = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=1,batch_first=True)

        layers = []
        for _ in range(opt.depth): # depth
            layers.extend([
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU()
            ])
        layers.append(nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1))
        self.output_projector = nn.Sequential(*layers)

        self.output_projector = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1),
        )

    def forward(self, x, c, t):
        hid_enc, _ = self.input_projector(x)         
        time_emb = time_embedding(t, self.hidden_dim, self.seq_len, self.device)
        h = hid_enc + time_emb                           
        out = self.output_projector(h.permute(0,2,1))
        return out.permute(0,2,1)                  
