'''
RNN exampel code
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets


# 하이퍼파라미터 정의
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 20
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
## colab GPU 사용 방법 --> 상단 메뉴에서 "런타임"클릭, "런타임 유형 변경"클릭, 하드웨어 가속기를 "GPU"로 변경 ##

# 데이터 로딩하기
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(trainset, min_freq=5)
LABEL.build_vocab(trainset)

# 학습용 데이터를 학습셋 80% 검증셋 20% 로 나누기
trainset, valset = trainset.split(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)
# 아래의 코드로 데이터를 직접 확인해 보세요.
#print(vars(trainset[0]))

vocab_size = len(TEXT.vocab)
n_classes = 2

class RNN(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(RNN, self).__init__()
        print("Building Basic RNN model...")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        ## 이부분을 GRU, LSTM으로 변경해 줄 수 있습니다. 
        ## 함수 사용 방법은 아래 링크를 참고해주세요
        ## GRU : https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        ## LSTM : https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.RNN(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _status = self.rnn(x, h_0)  # [i, b, h]

        # 예측을 위해 마지막 output만을 사용
        h_t = x[:,-1,:]
        self.dropout(h_t)
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

model = RNN(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[에폭: %d] 검증 오차:%5.2f | 검증 정확도:%5.2f" % (e, val_loss, val_accuracy))
    
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))

'''
[에폭: 1] 검증 오차: 0.69 | 검증 정확도:51.44
[에폭: 2] 검증 오차: 0.70 | 검증 정확도:50.20
[에폭: 3] 검증 오차: 0.70 | 검증 정확도:51.40
[에폭: 4] 검증 오차: 0.71 | 검증 정확도:50.70
[에폭: 5] 검증 오차: 0.73 | 검증 정확도:50.50
[에폭: 6] 검증 오차: 0.74 | 검증 정확도:50.78
[에폭: 7] 검증 오차: 0.75 | 검증 정확도:51.82
[에폭: 8] 검증 오차: 0.75 | 검증 정확도:51.04
[에폭: 9] 검증 오차: 0.74 | 검증 정확도:52.08
[에폭: 10] 검증 오차: 0.82 | 검증 정확도:58.84
[에폭: 11] 검증 오차: 0.56 | 검증 정확도:73.36
[에폭: 12] 검증 오차: 0.50 | 검증 정확도:78.16
[에폭: 13] 검증 오차: 0.49 | 검증 정확도:79.92
[에폭: 14] 검증 오차: 0.54 | 검증 정확도:79.82
[에폭: 15] 검증 오차: 0.54 | 검증 정확도:81.30
[에폭: 16] 검증 오차: 0.61 | 검증 정확도:79.98
[에폭: 17] 검증 오차: 0.56 | 검증 정확도:82.74
[에폭: 18] 검증 오차: 0.57 | 검증 정확도:81.22
[에폭: 19] 검증 오차: 0.59 | 검증 정확도:81.42
[에폭: 20] 검증 오차: 0.60 | 검증 정확도:81.70
테스트 오차:  0.51 | 테스트 정확도: 79.26
'''