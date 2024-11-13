### Векторное представление слов
### 2 words target: "завод" and "отрасль"

import gensim       #gensim is a library used to build deep learning models for natural language processing
import re           #re is a library that supports regular expressions for working with text strings

# load the word2vec model using KeyedVectors from the gensim library
word2vec = gensim.models.KeyedVectors.load_word2vec_format("/Users/vuhoanganh/Documents/семестр 7/cbow.txt", binary=False)

# define words (positive context) for which the model will find similar words
pos = ['предприятие_NOUN', 'промышленность_NOUN']

# return a list of 10 most similar words to those in pos (topn=10)
dist = word2vec.most_similar(positive=pos, topn=10)

# find words with the suffix "_NOUN" 
pat = re.compile("(.*)_NOUN")

for i in dist:
    print(i)
for i in dist:
    e = pat.match(i[0])
    if e is not None:
        print(e.group(1))

""" 
('отрасль_NOUN', 0.7817014455795288)
('производство_NOUN', 0.7622565031051636)
('промышленный_ADJ', 0.733397364616394)
('продукция_NOUN', 0.6384241580963135)
('машиностроение_NOUN', 0.6341099143028259)
('комбинат_NOUN', 0.6316563487052917)
('завод_NOUN', 0.6256354451179504)
('производственный_ADJ', 0.6253886222839355)
('индустрия_NOUN', 0.6100884079933167)
('подотрасль_NOUN', 0.5892025232315063)
отрасль
производство
продукция
машиностроение
комбинат
завод
индустрия
подотрасль 
"""