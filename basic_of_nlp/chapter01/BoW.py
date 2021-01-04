'''
BoW exampel code
'''

## no library
def bow(sentence):
  word_list = sentence.split(' ')
  word_list = list(set(word_list))
  embedding_matrix = [0 for element in range(len(word_list))]
  for index, word in enumerate(word_list):
    embedding_matrix[index] = sentence.count(word)
  return word_list, embedding_matrix

sentence = "Suzy is very very pretty woman and YoonA is very pretty woman too"
word_list, bow_embedding = bow(sentence)
print("word_list : ",word_list,", embedding : ",bow_embedding)

'''
word_list :  ['Suzy', 'pretty', 'very', 'YoonA', 'too', 'is', 'and', 'woman'] , embedding :  [1, 2, 3, 1, 1, 2, 1, 2]
'''

## using sklearn
from sklearn.feature_extraction.text import CountVectorizer

sentence = ["Suzy is very very pretty woman and YoonA is very pretty woman too"]
vectorizer = CountVectorizer(min_df = 1, ngram_range = (1,1))
embedding = vectorizer.fit_transform(sentence)
vocab = vectorizer.get_feature_names()
print("word_list : ",vocab,", embedding : ",embedding.toarray())

'''
word_list :  ['and', 'is', 'pretty', 'suzy', 'too', 'very', 'woman', 'yoona'] , embedding :  [[1 2 2 1 1 3 2 1]]
'''