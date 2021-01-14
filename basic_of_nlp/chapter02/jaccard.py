'''
Jaccard exampel code
'''
def jaccard(sent1,sent2):
  sent1_tokens = set(sent1.split(' '))
  sent2_tokens = set(sent2.split(' '))
  union = len(sent1_tokens | sent2_tokens)
  intersection = len(sent1_tokens & sent2_tokens)
  return intersection / union

  
sent1 = "i am so sorry but i love you"
sent2 = "i love you but i hate you"

result = jaccard(sent1,sent2)
print("Jaccard Similarity : ",result)

'''
Jaccard Similarity :  0.5
'''