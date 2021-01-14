'''
Euclidean exampel code
'''
from sentence_transformers import SentenceTransformer
import numpy as np

sent1 = "i am so sorry but i love you"
sent2 = "i love you but i hate you"
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def euclidean(sent1, sent2):
  sentences = [sent1,sent2]
  sentence_embeddings = model.encode(sentences)
  return np.sqrt(np.sum((sentence_embeddings[0]-sentence_embeddings[1])**2))

result = euclidean(sent1,sent2)
print("Euclidean Similarity : ",result)

'''
Euclidean Similarity :  4.518221
'''