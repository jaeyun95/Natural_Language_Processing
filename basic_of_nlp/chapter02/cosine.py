'''
Cosine exampel code
'''
from sentence_transformers import SentenceTransformer
import numpy as np

sent1 = "i am so sorry but i love you"
sent2 = "i love you but i hate you"
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def cosine(sent1, sent2):
  sentences = [sent1,sent2]
  sentence_embeddings = model.encode(sentences)
  return np.dot(sentence_embeddings[0],sentence_embeddings[1])/(np.linalg.norm(sentence_embeddings[0])*np.linalg.norm(sentence_embeddings[1]))

result = cosine(sent1,sent2)
print("Cosine Similarity : ",result)

'''
Cosine Similarity :  0.77774066
'''