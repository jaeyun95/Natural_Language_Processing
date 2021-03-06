"""
logistic regression
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

PATH = './data'
TEXT = data.Field(sequential=True, use_vocab=True, tokenize=str.split, lower=True, batch_first=True, fix_length=20)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=False, is_target=True)
train_data, test_data = TabularDataset.splits(
        path=PATH, train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

train_loader = Iterator(dataset=train_data, batch_size = 5)
test_loader = Iterator(dataset=test_data, batch_size = 5)
batch = next(iter(train_loader))

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, input_data):
        x = input_data.type(torch.FloatTensor)
        x = self.fc1(x)
        x = F.relu(x)
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

model = LogisticRegression(input_dim=20,output_dim=2)
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(0,30):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model,test_loader)
    print('[{}] Test Loss : {:.4f}, Accuracy : {:.2f}%'.format(epoch, test_loss, test_accuracy))