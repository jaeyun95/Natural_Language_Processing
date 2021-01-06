'''
Skip-Gram exampel code
'''

## using pytorch
import torch
import torch.nn as nn

EMBEDDING_DIM = 128
EPOCHS = 200
CONTEXT_SIZE = 4

example_sentence = """In the case of CBOW, one word is eliminated, and the word is predicted from surrounding words.
Therefore, it takes multiple input vectors as inputs to the model and creates one output vector.
In contrast, Skip-Gram learns by removing all words except one word and predicting the surrounding words in the context through one word. 
So, it takes a vector as input and produces multiple output vectors.
CBOW and Skip-Gram are different.""".split()

# convert context to index vector
def make_context_vector(context, word_to_ix):
  idxs = word_to_ix[context]
  return torch.tensor(idxs, dtype=torch.long)

# make dataset function
def make_data(sentence):
  data = []
  for i in range(2, len(example_sentence) - 2):
    context = example_sentence[i]
    target = [example_sentence[i - 2], example_sentence[i - 1], example_sentence[i + 1], example_sentence[i + 2]]
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
class SKIP_GRAM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size):
    super(SKIP_GRAM, self).__init__()
    self.context_size = context_size
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    self.layer1 = nn.Linear(embedding_dim, 64)
    self.activation1 = nn.ReLU()

    self.layer2 = nn.Linear(64, vocab_size * context_size)
    self.activation2 = nn.LogSoftmax(dim = -1)

  def forward(self, inputs):
    embeded_vector = self.embeddings(inputs)
    output = self.activation1(self.layer1(embeded_vector))
    output = self.activation2(self.layer2(output))
    return output.view(self.context_size,vocab_size)

model = SKIP_GRAM(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training
for epoch in range(EPOCHS):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_index)  
        log_probs = model(context_vector)
        total_loss += loss_function(log_probs, torch.tensor([word_to_index[t] for t in target]))
    print('epoch = ',epoch, ', loss = ',total_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# test
test_data = 'Skip-Gram'
test_vector = make_context_vector(test_data, word_to_index)
result = model(test_vector)
print('Prediction : ', [index_to_word[torch.argmax(r).item()] for r in result])

'''
epoch =  0 , loss =  tensor(388.8869, grad_fn=<AddBackward0>)
epoch =  1 , loss =  tensor(388.2437, grad_fn=<AddBackward0>)
epoch =  2 , loss =  tensor(387.6069, grad_fn=<AddBackward0>)
epoch =  3 , loss =  tensor(386.9760, grad_fn=<AddBackward0>)
epoch =  4 , loss =  tensor(386.3522, grad_fn=<AddBackward0>)
epoch =  5 , loss =  tensor(385.7342, grad_fn=<AddBackward0>)
epoch =  6 , loss =  tensor(385.1223, grad_fn=<AddBackward0>)
epoch =  7 , loss =  tensor(384.5179, grad_fn=<AddBackward0>)
epoch =  8 , loss =  tensor(383.9195, grad_fn=<AddBackward0>)
epoch =  9 , loss =  tensor(383.3261, grad_fn=<AddBackward0>)
epoch =  10 , loss =  tensor(382.7373, grad_fn=<AddBackward0>)
epoch =  11 , loss =  tensor(382.1507, grad_fn=<AddBackward0>)
epoch =  12 , loss =  tensor(381.5667, grad_fn=<AddBackward0>)
epoch =  13 , loss =  tensor(380.9860, grad_fn=<AddBackward0>)
epoch =  14 , loss =  tensor(380.4091, grad_fn=<AddBackward0>)
epoch =  15 , loss =  tensor(379.8357, grad_fn=<AddBackward0>)
epoch =  16 , loss =  tensor(379.2690, grad_fn=<AddBackward0>)
epoch =  17 , loss =  tensor(378.7043, grad_fn=<AddBackward0>)
epoch =  18 , loss =  tensor(378.1423, grad_fn=<AddBackward0>)
epoch =  19 , loss =  tensor(377.5824, grad_fn=<AddBackward0>)
epoch =  20 , loss =  tensor(377.0222, grad_fn=<AddBackward0>)
epoch =  21 , loss =  tensor(376.4612, grad_fn=<AddBackward0>)
epoch =  22 , loss =  tensor(375.9008, grad_fn=<AddBackward0>)
epoch =  23 , loss =  tensor(375.3377, grad_fn=<AddBackward0>)
epoch =  24 , loss =  tensor(374.7762, grad_fn=<AddBackward0>)
epoch =  25 , loss =  tensor(374.2155, grad_fn=<AddBackward0>)
epoch =  26 , loss =  tensor(373.6546, grad_fn=<AddBackward0>)
epoch =  27 , loss =  tensor(373.0938, grad_fn=<AddBackward0>)
epoch =  28 , loss =  tensor(372.5332, grad_fn=<AddBackward0>)
epoch =  29 , loss =  tensor(371.9720, grad_fn=<AddBackward0>)
epoch =  30 , loss =  tensor(371.4123, grad_fn=<AddBackward0>)
epoch =  31 , loss =  tensor(370.8505, grad_fn=<AddBackward0>)
epoch =  32 , loss =  tensor(370.2869, grad_fn=<AddBackward0>)
epoch =  33 , loss =  tensor(369.7224, grad_fn=<AddBackward0>)
epoch =  34 , loss =  tensor(369.1603, grad_fn=<AddBackward0>)
epoch =  35 , loss =  tensor(368.5994, grad_fn=<AddBackward0>)
epoch =  36 , loss =  tensor(368.0379, grad_fn=<AddBackward0>)
epoch =  37 , loss =  tensor(367.4742, grad_fn=<AddBackward0>)
epoch =  38 , loss =  tensor(366.9069, grad_fn=<AddBackward0>)
epoch =  39 , loss =  tensor(366.3350, grad_fn=<AddBackward0>)
epoch =  40 , loss =  tensor(365.7593, grad_fn=<AddBackward0>)
epoch =  41 , loss =  tensor(365.1814, grad_fn=<AddBackward0>)
epoch =  42 , loss =  tensor(364.6024, grad_fn=<AddBackward0>)
epoch =  43 , loss =  tensor(364.0194, grad_fn=<AddBackward0>)
epoch =  44 , loss =  tensor(363.4347, grad_fn=<AddBackward0>)
epoch =  45 , loss =  tensor(362.8495, grad_fn=<AddBackward0>)
epoch =  46 , loss =  tensor(362.2609, grad_fn=<AddBackward0>)
epoch =  47 , loss =  tensor(361.6683, grad_fn=<AddBackward0>)
epoch =  48 , loss =  tensor(361.0721, grad_fn=<AddBackward0>)
epoch =  49 , loss =  tensor(360.4712, grad_fn=<AddBackward0>)
epoch =  50 , loss =  tensor(359.8673, grad_fn=<AddBackward0>)
epoch =  51 , loss =  tensor(359.2595, grad_fn=<AddBackward0>)
epoch =  52 , loss =  tensor(358.6474, grad_fn=<AddBackward0>)
epoch =  53 , loss =  tensor(358.0304, grad_fn=<AddBackward0>)
epoch =  54 , loss =  tensor(357.4091, grad_fn=<AddBackward0>)
epoch =  55 , loss =  tensor(356.7816, grad_fn=<AddBackward0>)
epoch =  56 , loss =  tensor(356.1496, grad_fn=<AddBackward0>)
epoch =  57 , loss =  tensor(355.5130, grad_fn=<AddBackward0>)
epoch =  58 , loss =  tensor(354.8698, grad_fn=<AddBackward0>)
epoch =  59 , loss =  tensor(354.2195, grad_fn=<AddBackward0>)
epoch =  60 , loss =  tensor(353.5646, grad_fn=<AddBackward0>)
epoch =  61 , loss =  tensor(352.9039, grad_fn=<AddBackward0>)
epoch =  62 , loss =  tensor(352.2380, grad_fn=<AddBackward0>)
epoch =  63 , loss =  tensor(351.5661, grad_fn=<AddBackward0>)
epoch =  64 , loss =  tensor(350.8868, grad_fn=<AddBackward0>)
epoch =  65 , loss =  tensor(350.2021, grad_fn=<AddBackward0>)
epoch =  66 , loss =  tensor(349.5116, grad_fn=<AddBackward0>)
epoch =  67 , loss =  tensor(348.8150, grad_fn=<AddBackward0>)
epoch =  68 , loss =  tensor(348.1128, grad_fn=<AddBackward0>)
epoch =  69 , loss =  tensor(347.4050, grad_fn=<AddBackward0>)
epoch =  70 , loss =  tensor(346.6921, grad_fn=<AddBackward0>)
epoch =  71 , loss =  tensor(345.9731, grad_fn=<AddBackward0>)
epoch =  72 , loss =  tensor(345.2487, grad_fn=<AddBackward0>)
epoch =  73 , loss =  tensor(344.5190, grad_fn=<AddBackward0>)
epoch =  74 , loss =  tensor(343.7846, grad_fn=<AddBackward0>)
epoch =  75 , loss =  tensor(343.0439, grad_fn=<AddBackward0>)
epoch =  76 , loss =  tensor(342.2986, grad_fn=<AddBackward0>)
epoch =  77 , loss =  tensor(341.5461, grad_fn=<AddBackward0>)
epoch =  78 , loss =  tensor(340.7880, grad_fn=<AddBackward0>)
epoch =  79 , loss =  tensor(340.0252, grad_fn=<AddBackward0>)
epoch =  80 , loss =  tensor(339.2560, grad_fn=<AddBackward0>)
epoch =  81 , loss =  tensor(338.4823, grad_fn=<AddBackward0>)
epoch =  82 , loss =  tensor(337.7029, grad_fn=<AddBackward0>)
epoch =  83 , loss =  tensor(336.9173, grad_fn=<AddBackward0>)
epoch =  84 , loss =  tensor(336.1272, grad_fn=<AddBackward0>)
epoch =  85 , loss =  tensor(335.3311, grad_fn=<AddBackward0>)
epoch =  86 , loss =  tensor(334.5300, grad_fn=<AddBackward0>)
epoch =  87 , loss =  tensor(333.7233, grad_fn=<AddBackward0>)
epoch =  88 , loss =  tensor(332.9114, grad_fn=<AddBackward0>)
epoch =  89 , loss =  tensor(332.0950, grad_fn=<AddBackward0>)
epoch =  90 , loss =  tensor(331.2734, grad_fn=<AddBackward0>)
epoch =  91 , loss =  tensor(330.4472, grad_fn=<AddBackward0>)
epoch =  92 , loss =  tensor(329.6164, grad_fn=<AddBackward0>)
epoch =  93 , loss =  tensor(328.7796, grad_fn=<AddBackward0>)
epoch =  94 , loss =  tensor(327.9379, grad_fn=<AddBackward0>)
epoch =  95 , loss =  tensor(327.0900, grad_fn=<AddBackward0>)
epoch =  96 , loss =  tensor(326.2383, grad_fn=<AddBackward0>)
epoch =  97 , loss =  tensor(325.3820, grad_fn=<AddBackward0>)
epoch =  98 , loss =  tensor(324.5188, grad_fn=<AddBackward0>)
epoch =  99 , loss =  tensor(323.6453, grad_fn=<AddBackward0>)
epoch =  100 , loss =  tensor(322.7663, grad_fn=<AddBackward0>)
epoch =  101 , loss =  tensor(321.8809, grad_fn=<AddBackward0>)
epoch =  102 , loss =  tensor(320.9919, grad_fn=<AddBackward0>)
epoch =  103 , loss =  tensor(320.0972, grad_fn=<AddBackward0>)
epoch =  104 , loss =  tensor(319.1995, grad_fn=<AddBackward0>)
epoch =  105 , loss =  tensor(318.2959, grad_fn=<AddBackward0>)
epoch =  106 , loss =  tensor(317.3880, grad_fn=<AddBackward0>)
epoch =  107 , loss =  tensor(316.4772, grad_fn=<AddBackward0>)
epoch =  108 , loss =  tensor(315.5608, grad_fn=<AddBackward0>)
epoch =  109 , loss =  tensor(314.6396, grad_fn=<AddBackward0>)
epoch =  110 , loss =  tensor(313.7152, grad_fn=<AddBackward0>)
epoch =  111 , loss =  tensor(312.7863, grad_fn=<AddBackward0>)
epoch =  112 , loss =  tensor(311.8554, grad_fn=<AddBackward0>)
epoch =  113 , loss =  tensor(310.9204, grad_fn=<AddBackward0>)
epoch =  114 , loss =  tensor(309.9818, grad_fn=<AddBackward0>)
epoch =  115 , loss =  tensor(309.0414, grad_fn=<AddBackward0>)
epoch =  116 , loss =  tensor(308.0978, grad_fn=<AddBackward0>)
epoch =  117 , loss =  tensor(307.1506, grad_fn=<AddBackward0>)
epoch =  118 , loss =  tensor(306.1999, grad_fn=<AddBackward0>)
epoch =  119 , loss =  tensor(305.2460, grad_fn=<AddBackward0>)
epoch =  120 , loss =  tensor(304.2907, grad_fn=<AddBackward0>)
epoch =  121 , loss =  tensor(303.3316, grad_fn=<AddBackward0>)
epoch =  122 , loss =  tensor(302.3695, grad_fn=<AddBackward0>)
epoch =  123 , loss =  tensor(301.4060, grad_fn=<AddBackward0>)
epoch =  124 , loss =  tensor(300.4371, grad_fn=<AddBackward0>)
epoch =  125 , loss =  tensor(299.4684, grad_fn=<AddBackward0>)
epoch =  126 , loss =  tensor(298.4967, grad_fn=<AddBackward0>)
epoch =  127 , loss =  tensor(297.5236, grad_fn=<AddBackward0>)
epoch =  128 , loss =  tensor(296.5450, grad_fn=<AddBackward0>)
epoch =  129 , loss =  tensor(295.5650, grad_fn=<AddBackward0>)
epoch =  130 , loss =  tensor(294.5836, grad_fn=<AddBackward0>)
epoch =  131 , loss =  tensor(293.6015, grad_fn=<AddBackward0>)
epoch =  132 , loss =  tensor(292.6165, grad_fn=<AddBackward0>)
epoch =  133 , loss =  tensor(291.6292, grad_fn=<AddBackward0>)
epoch =  134 , loss =  tensor(290.6412, grad_fn=<AddBackward0>)
epoch =  135 , loss =  tensor(289.6526, grad_fn=<AddBackward0>)
epoch =  136 , loss =  tensor(288.6598, grad_fn=<AddBackward0>)
epoch =  137 , loss =  tensor(287.6668, grad_fn=<AddBackward0>)
epoch =  138 , loss =  tensor(286.6717, grad_fn=<AddBackward0>)
epoch =  139 , loss =  tensor(285.6774, grad_fn=<AddBackward0>)
epoch =  140 , loss =  tensor(284.6801, grad_fn=<AddBackward0>)
epoch =  141 , loss =  tensor(283.6810, grad_fn=<AddBackward0>)
epoch =  142 , loss =  tensor(282.6843, grad_fn=<AddBackward0>)
epoch =  143 , loss =  tensor(281.6855, grad_fn=<AddBackward0>)
epoch =  144 , loss =  tensor(280.6855, grad_fn=<AddBackward0>)
epoch =  145 , loss =  tensor(279.6846, grad_fn=<AddBackward0>)
epoch =  146 , loss =  tensor(278.6856, grad_fn=<AddBackward0>)
epoch =  147 , loss =  tensor(277.6845, grad_fn=<AddBackward0>)
epoch =  148 , loss =  tensor(276.6833, grad_fn=<AddBackward0>)
epoch =  149 , loss =  tensor(275.6823, grad_fn=<AddBackward0>)
epoch =  150 , loss =  tensor(274.6802, grad_fn=<AddBackward0>)
epoch =  151 , loss =  tensor(273.6772, grad_fn=<AddBackward0>)
epoch =  152 , loss =  tensor(272.6748, grad_fn=<AddBackward0>)
epoch =  153 , loss =  tensor(271.6737, grad_fn=<AddBackward0>)
epoch =  154 , loss =  tensor(270.6734, grad_fn=<AddBackward0>)
epoch =  155 , loss =  tensor(269.6714, grad_fn=<AddBackward0>)
epoch =  156 , loss =  tensor(268.6700, grad_fn=<AddBackward0>)
epoch =  157 , loss =  tensor(267.6691, grad_fn=<AddBackward0>)
epoch =  158 , loss =  tensor(266.6684, grad_fn=<AddBackward0>)
epoch =  159 , loss =  tensor(265.6671, grad_fn=<AddBackward0>)
epoch =  160 , loss =  tensor(264.6697, grad_fn=<AddBackward0>)
epoch =  161 , loss =  tensor(263.6678, grad_fn=<AddBackward0>)
epoch =  162 , loss =  tensor(262.6686, grad_fn=<AddBackward0>)
epoch =  163 , loss =  tensor(261.6700, grad_fn=<AddBackward0>)
epoch =  164 , loss =  tensor(260.6720, grad_fn=<AddBackward0>)
epoch =  165 , loss =  tensor(259.6743, grad_fn=<AddBackward0>)
epoch =  166 , loss =  tensor(258.6800, grad_fn=<AddBackward0>)
epoch =  167 , loss =  tensor(257.6846, grad_fn=<AddBackward0>)
epoch =  168 , loss =  tensor(256.6918, grad_fn=<AddBackward0>)
epoch =  169 , loss =  tensor(255.7001, grad_fn=<AddBackward0>)
epoch =  170 , loss =  tensor(254.7110, grad_fn=<AddBackward0>)
epoch =  171 , loss =  tensor(253.7228, grad_fn=<AddBackward0>)
epoch =  172 , loss =  tensor(252.7360, grad_fn=<AddBackward0>)
epoch =  173 , loss =  tensor(251.7511, grad_fn=<AddBackward0>)
epoch =  174 , loss =  tensor(250.7663, grad_fn=<AddBackward0>)
epoch =  175 , loss =  tensor(249.7831, grad_fn=<AddBackward0>)
epoch =  176 , loss =  tensor(248.8033, grad_fn=<AddBackward0>)
epoch =  177 , loss =  tensor(247.8258, grad_fn=<AddBackward0>)
epoch =  178 , loss =  tensor(246.8494, grad_fn=<AddBackward0>)
epoch =  179 , loss =  tensor(245.8760, grad_fn=<AddBackward0>)
epoch =  180 , loss =  tensor(244.9057, grad_fn=<AddBackward0>)
epoch =  181 , loss =  tensor(243.9388, grad_fn=<AddBackward0>)
epoch =  182 , loss =  tensor(242.9735, grad_fn=<AddBackward0>)
epoch =  183 , loss =  tensor(242.0119, grad_fn=<AddBackward0>)
epoch =  184 , loss =  tensor(241.0523, grad_fn=<AddBackward0>)
epoch =  185 , loss =  tensor(240.0972, grad_fn=<AddBackward0>)
epoch =  186 , loss =  tensor(239.1439, grad_fn=<AddBackward0>)
epoch =  187 , loss =  tensor(238.1925, grad_fn=<AddBackward0>)
epoch =  188 , loss =  tensor(237.2450, grad_fn=<AddBackward0>)
epoch =  189 , loss =  tensor(236.2995, grad_fn=<AddBackward0>)
epoch =  190 , loss =  tensor(235.3596, grad_fn=<AddBackward0>)
epoch =  191 , loss =  tensor(234.4232, grad_fn=<AddBackward0>)
epoch =  192 , loss =  tensor(233.4901, grad_fn=<AddBackward0>)
epoch =  193 , loss =  tensor(232.5601, grad_fn=<AddBackward0>)
epoch =  194 , loss =  tensor(231.6366, grad_fn=<AddBackward0>)
epoch =  195 , loss =  tensor(230.7154, grad_fn=<AddBackward0>)
epoch =  196 , loss =  tensor(229.7982, grad_fn=<AddBackward0>)
epoch =  197 , loss =  tensor(228.8860, grad_fn=<AddBackward0>)
epoch =  198 , loss =  tensor(227.9756, grad_fn=<AddBackward0>)
epoch =  199 , loss =  tensor(227.0694, grad_fn=<AddBackward0>)
Prediction :  ['CBOW', 'and', 'are', 'by']
'''