### Векторное представление слов
### 2 words: "завод" and "отрасль"

import gensim       
import re

pat = re.compile("(.*)_NOUN")

word2vec = gensim.models.KeyedVectors.load_word2vec_format("/Users/vuhoanganh/Documents/семестр 7/Обработка естественного языка/NLP/Lab2/cbow.txt", binary=False)

pos = ["завод_NOUN", "отрасль_NOUN"]

dist = word2vec.most_similar(positive=pos, topn=10)
for i in dist:
    print(i)
for i in dist:
    e = pat.match(i[0])
    if e is not None:
        print(e.group(1))


# result:
# ('предприятие_NOUN', 0.7937479615211487)
# ('промышленность_NOUN', 0.775575578212738)
# ('производство_NOUN', 0.7192516922950745)
# ('комбинат_NOUN', 0.6872789859771729)
# ('промышленный_ADJ', 0.6495099067687988)
# ('подотрасль_NOUN', 0.6494302153587341)
# ('машиностроительный_ADJ', 0.6254802942276001)
# ('машиностроение_NOUN', 0.6004259586334229)
# ('нефтепереработка_NOUN', 0.5961022973060608)
# ('автозавод_NOUN', 0.5927801728248596)
# предприятие
# промышленность
# производство
# комбинат
# подотрасль
# машиностроение
# нефтепереработка
# автозавод