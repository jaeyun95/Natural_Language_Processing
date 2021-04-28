"""
GRU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data 
from torchtext.data import TabularDataset
import urllib.request
import pandas as pd
from torchtext.data import Iterator

PATH = './'
TEXT = data.Field(sequential=True, use_vocab=True, tokenize=str.split, lower=True, batch_first=True, fix_length=20)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=False, is_target=True)
train_data, test_data = TabularDataset.splits(
        path=PATH, train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

train_loader = Iterator(dataset=train_data, batch_size = 5)
test_loader = Iterator(dataset=test_data, batch_size = 5)
batch = next(iter(train_loader))

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) 
        x, _ = self.gru(x, h_0) 
        h_t = x[:,-1,:] 
        self.dropout(h_t)
        logit = self.out(h_t)  
        prob = F.softmax(logit)
        return prob
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
    
def train(model, train_loader, optimizer):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
def evaluate(model, test_data):
    model.eval()
    corrects, total_loss = 0, 0
    for batch_index, (data, target) in enumerate(test_loader):
        logit = model(data)
        loss = F.cross_entropy(logit, target, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(target.size()).data == target.data).sum()
    size = len(test_loader.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

model = GRU(1,256,len(TEXT.vocab),128, 2, 0.5)
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(0,30):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model,test_loader)
    print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))