'''
one hot encoding example code
'''

## no library
def one_hot(word_list):
  word_list = list(set(word_list))
  encoding_matrix = [[0 for col in range(len(word_list))] for row in range(len(word_list))]
  for index, word in enumerate(word_list):
    encoding_matrix[index][index] = 1
  return encoding_matrix

labels = ['cat','dog','rabbit','turtle']

'''
label :  cat , encoding :  [1, 0, 0, 0]
label :  dog , encoding :  [0, 1, 0, 0]
label :  rabbit , encoding :  [0, 0, 1, 0]
label :  turtle , encoding :  [0, 0, 0, 1]
'''

## using pandas
import pandas as pd

label_dict = {'label':['cat','dog','rabbit','turtle']}
#df = pd.DataFrame(label_dict)
one_hot_encoding = pd.get_dummies(label_dict['label'])
print(one_hot_encoding)

'''
   cat  dog  rabbit  turtle
0    1    0       0       0
1    0    1       0       0
2    0    0       1       0
3    0    0       0       1
'''

## using sklearn
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

label_dict = {'label':['cat','dog','rabbit','turtle']}
df = pd.DataFrame(label_dict)
one_hot = OneHotEncoder()
one_hot_encoding = one_hot.fit_transform(df)
print(one_hot_encoding)

'''
(0, 0)	1.0
(1, 1)	1.0
(2, 2)	1.0
(3, 3)	1.0
'''