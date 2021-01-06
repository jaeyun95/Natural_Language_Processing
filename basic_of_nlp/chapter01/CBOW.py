'''
CBOW exampel code
'''

## using pytorch
import torch
import torch.nn as nn

EMBEDDING_DIM = 128
EPOCHS = 100

example_sentence = """In the case of CBOW, one word is eliminated, and the word is predicted from surrounding words.
Therefore, it takes multiple input vectors as inputs to the model and creates one output vector.
In contrast, Skip-Gram learns by removing all words except one word and predicting the surrounding words in the context through one word. 
So, it takes a vector as input and produces multiple output vectors.
CBOW and Skip-Gram are different.""".split()

# convert context to index vector
def make_context_vector(context, word_to_ix):
  idxs = [word_to_ix[w] for w in context]
  return torch.tensor(idxs, dtype=torch.long)

# make dataset function
def make_data(sentence):
  data = []
  for i in range(2, len(example_sentence) - 2):
    context = [example_sentence[i - 2], example_sentence[i - 1], example_sentence[i + 1], example_sentence[i + 2]]
    target = example_sentence[i]
    data.append((context, target))
  return data

# make vocasb and vocab size
vocab = set(example_sentence)
vocab_size = len(example_sentence)

# make index, word dictionary
word_to_index = {word:index for index, word in enumerate(vocab)}
index_to_word = {index:word for index, word in enumerate(vocab)}

# make training data
data = make_data(example_sentence)

# model defination
class CBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    self.layer1 = nn.Linear(embedding_dim, 64)
    self.activation1 = nn.ReLU()

    self.layer2 = nn.Linear(64, vocab_size)
    self.activation2 = nn.LogSoftmax(dim = -1)

  def forward(self, inputs):
    embeded_vector = sum(self.embeddings(inputs)).view(1,-1)
    output = self.activation1(self.layer1(embeded_vector))
    output = self.activation2(self.layer2(output))
    return output

model = CBOW(vocab_size, EMBEDDING_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training
for epoch in range(EPOCHS):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_index)  
        log_probs = model(context_vector)
        total_loss += loss_function(log_probs, torch.tensor([word_to_index[target]]))
    print('epoch = ',epoch, ', loss = ',total_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# test
test_data = ['CBOW','and','are','different.']
test_vector = make_context_vector(test_data, word_to_index)
result = model(test_vector)
print('Prediction : ', index_to_word[torch.argmax(result[0]).item()])

'''
epoch =  0 , loss =  tensor(301.5551, grad_fn=<AddBackward0>)
epoch =  1 , loss =  tensor(292.3177, grad_fn=<AddBackward0>)
epoch =  2 , loss =  tensor(283.8391, grad_fn=<AddBackward0>)
epoch =  3 , loss =  tensor(275.7932, grad_fn=<AddBackward0>)
epoch =  4 , loss =  tensor(267.9984, grad_fn=<AddBackward0>)
epoch =  5 , loss =  tensor(260.4667, grad_fn=<AddBackward0>)
epoch =  6 , loss =  tensor(253.0328, grad_fn=<AddBackward0>)
epoch =  7 , loss =  tensor(245.6383, grad_fn=<AddBackward0>)
epoch =  8 , loss =  tensor(238.1835, grad_fn=<AddBackward0>)
epoch =  9 , loss =  tensor(230.6743, grad_fn=<AddBackward0>)
epoch =  10 , loss =  tensor(223.1495, grad_fn=<AddBackward0>)
epoch =  11 , loss =  tensor(215.5965, grad_fn=<AddBackward0>)
epoch =  12 , loss =  tensor(208.0219, grad_fn=<AddBackward0>)
epoch =  13 , loss =  tensor(200.4068, grad_fn=<AddBackward0>)
epoch =  14 , loss =  tensor(192.7762, grad_fn=<AddBackward0>)
epoch =  15 , loss =  tensor(185.1789, grad_fn=<AddBackward0>)
epoch =  16 , loss =  tensor(177.6059, grad_fn=<AddBackward0>)
epoch =  17 , loss =  tensor(170.0684, grad_fn=<AddBackward0>)
epoch =  18 , loss =  tensor(162.6431, grad_fn=<AddBackward0>)
epoch =  19 , loss =  tensor(155.2471, grad_fn=<AddBackward0>)
epoch =  20 , loss =  tensor(147.9686, grad_fn=<AddBackward0>)
epoch =  21 , loss =  tensor(140.8398, grad_fn=<AddBackward0>)
epoch =  22 , loss =  tensor(133.8264, grad_fn=<AddBackward0>)
epoch =  23 , loss =  tensor(126.9840, grad_fn=<AddBackward0>)
epoch =  24 , loss =  tensor(120.2801, grad_fn=<AddBackward0>)
epoch =  25 , loss =  tensor(113.7605, grad_fn=<AddBackward0>)
epoch =  26 , loss =  tensor(107.4410, grad_fn=<AddBackward0>)
epoch =  27 , loss =  tensor(101.3456, grad_fn=<AddBackward0>)
epoch =  28 , loss =  tensor(95.5053, grad_fn=<AddBackward0>)
epoch =  29 , loss =  tensor(89.9347, grad_fn=<AddBackward0>)
epoch =  30 , loss =  tensor(84.6207, grad_fn=<AddBackward0>)
epoch =  31 , loss =  tensor(79.5470, grad_fn=<AddBackward0>)
epoch =  32 , loss =  tensor(74.7555, grad_fn=<AddBackward0>)
epoch =  33 , loss =  tensor(70.2152, grad_fn=<AddBackward0>)
epoch =  34 , loss =  tensor(65.9483, grad_fn=<AddBackward0>)
epoch =  35 , loss =  tensor(61.9185, grad_fn=<AddBackward0>)
epoch =  36 , loss =  tensor(58.1479, grad_fn=<AddBackward0>)
epoch =  37 , loss =  tensor(54.6099, grad_fn=<AddBackward0>)
epoch =  38 , loss =  tensor(51.3009, grad_fn=<AddBackward0>)
epoch =  39 , loss =  tensor(48.2249, grad_fn=<AddBackward0>)
epoch =  40 , loss =  tensor(45.3639, grad_fn=<AddBackward0>)
epoch =  41 , loss =  tensor(42.6915, grad_fn=<AddBackward0>)
epoch =  42 , loss =  tensor(40.2216, grad_fn=<AddBackward0>)
epoch =  43 , loss =  tensor(37.9330, grad_fn=<AddBackward0>)
epoch =  44 , loss =  tensor(35.8202, grad_fn=<AddBackward0>)
epoch =  45 , loss =  tensor(33.8514, grad_fn=<AddBackward0>)
epoch =  46 , loss =  tensor(32.0346, grad_fn=<AddBackward0>)
epoch =  47 , loss =  tensor(30.3560, grad_fn=<AddBackward0>)
epoch =  48 , loss =  tensor(28.7877, grad_fn=<AddBackward0>)
epoch =  49 , loss =  tensor(27.3420, grad_fn=<AddBackward0>)
epoch =  50 , loss =  tensor(25.9963, grad_fn=<AddBackward0>)
epoch =  51 , loss =  tensor(24.7508, grad_fn=<AddBackward0>)
epoch =  52 , loss =  tensor(23.5891, grad_fn=<AddBackward0>)
epoch =  53 , loss =  tensor(22.5082, grad_fn=<AddBackward0>)
epoch =  54 , loss =  tensor(21.5012, grad_fn=<AddBackward0>)
epoch =  55 , loss =  tensor(20.5618, grad_fn=<AddBackward0>)
epoch =  56 , loss =  tensor(19.6836, grad_fn=<AddBackward0>)
epoch =  57 , loss =  tensor(18.8640, grad_fn=<AddBackward0>)
epoch =  58 , loss =  tensor(18.0991, grad_fn=<AddBackward0>)
epoch =  59 , loss =  tensor(17.3841, grad_fn=<AddBackward0>)
epoch =  60 , loss =  tensor(16.7121, grad_fn=<AddBackward0>)
epoch =  61 , loss =  tensor(16.0841, grad_fn=<AddBackward0>)
epoch =  62 , loss =  tensor(15.4941, grad_fn=<AddBackward0>)
epoch =  63 , loss =  tensor(14.9383, grad_fn=<AddBackward0>)
epoch =  64 , loss =  tensor(14.4161, grad_fn=<AddBackward0>)
epoch =  65 , loss =  tensor(13.9240, grad_fn=<AddBackward0>)
epoch =  66 , loss =  tensor(13.4583, grad_fn=<AddBackward0>)
epoch =  67 , loss =  tensor(13.0211, grad_fn=<AddBackward0>)
epoch =  68 , loss =  tensor(12.6057, grad_fn=<AddBackward0>)
epoch =  69 , loss =  tensor(12.2119, grad_fn=<AddBackward0>)
epoch =  70 , loss =  tensor(11.8408, grad_fn=<AddBackward0>)
epoch =  71 , loss =  tensor(11.4876, grad_fn=<AddBackward0>)
epoch =  72 , loss =  tensor(11.1513, grad_fn=<AddBackward0>)
epoch =  73 , loss =  tensor(10.8323, grad_fn=<AddBackward0>)
epoch =  74 , loss =  tensor(10.5284, grad_fn=<AddBackward0>)
epoch =  75 , loss =  tensor(10.2394, grad_fn=<AddBackward0>)
epoch =  76 , loss =  tensor(9.9630, grad_fn=<AddBackward0>)
epoch =  77 , loss =  tensor(9.7001, grad_fn=<AddBackward0>)
epoch =  78 , loss =  tensor(9.4486, grad_fn=<AddBackward0>)
epoch =  79 , loss =  tensor(9.2083, grad_fn=<AddBackward0>)
epoch =  80 , loss =  tensor(8.9787, grad_fn=<AddBackward0>)
epoch =  81 , loss =  tensor(8.7584, grad_fn=<AddBackward0>)
epoch =  82 , loss =  tensor(8.5485, grad_fn=<AddBackward0>)
epoch =  83 , loss =  tensor(8.3455, grad_fn=<AddBackward0>)
epoch =  84 , loss =  tensor(8.1522, grad_fn=<AddBackward0>)
epoch =  85 , loss =  tensor(7.9664, grad_fn=<AddBackward0>)
epoch =  86 , loss =  tensor(7.7876, grad_fn=<AddBackward0>)
epoch =  87 , loss =  tensor(7.6161, grad_fn=<AddBackward0>)
epoch =  88 , loss =  tensor(7.4511, grad_fn=<AddBackward0>)
epoch =  89 , loss =  tensor(7.2922, grad_fn=<AddBackward0>)
epoch =  90 , loss =  tensor(7.1388, grad_fn=<AddBackward0>)
epoch =  91 , loss =  tensor(6.9916, grad_fn=<AddBackward0>)
epoch =  92 , loss =  tensor(6.8496, grad_fn=<AddBackward0>)
epoch =  93 , loss =  tensor(6.7129, grad_fn=<AddBackward0>)
epoch =  94 , loss =  tensor(6.5809, grad_fn=<AddBackward0>)
epoch =  95 , loss =  tensor(6.4529, grad_fn=<AddBackward0>)
epoch =  96 , loss =  tensor(6.3301, grad_fn=<AddBackward0>)
epoch =  97 , loss =  tensor(6.2107, grad_fn=<AddBackward0>)
epoch =  98 , loss =  tensor(6.0960, grad_fn=<AddBackward0>)
epoch =  99 , loss =  tensor(5.9844, grad_fn=<AddBackward0>)
Prediction :  Skip-Gram
'''