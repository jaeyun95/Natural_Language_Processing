"""
CNN
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

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_vocab, embed_dim, dropout_p=0.2):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.conv1 = nn.Conv1d(self.input_dim,self.input_dim//2,kernel_size=2)
        self.pool = nn.MaxPool1d(4)
        self.drop = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(630,256)
        self.fc2 = nn.Linear(256,self.output_dim)
        
    def forward(self, x):
        x = self.embed(x)
        x = F.relu(self.pool(self.conv1(x)))
        x = self.drop(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x= self.drop(x)
        x = F.relu(self.fc2(x))
        prob = F.softmax(x)
        return prob
    
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

model = CNN(20,2,len(TEXT.vocab),128)
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(0,30):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model,test_loader)
    print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))