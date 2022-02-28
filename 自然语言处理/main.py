"""
自然语言理解技术 - 大作业
"""
data = '自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。' \
       '自然语言处理是一门融语言学、计算机科学、数学于一体的科学。2006年，杰弗里·辛顿以及他的学生鲁斯兰·萨拉赫丁诺夫正式提出了深度学习的概念。' \
       '随之，研究者开始把目光转向深度学习。'
# 分词
import jieba
wordlist = jieba.cut(data, cut_all=False)
print("/ ".join(wordlist))

# 词性标注
import jieba.posseg as psg
seg = psg.cut(data)
for ele in seg:
    print(ele)

import sys
sys.path.append('../')
import re
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

def pack_ws_pos_sentece(sentence_ws, sentence_pos):
    assert len(sentence_ws) == len(sentence_pos)
    res = []
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        res.append(f"{word_ws}({word_pos})")
    return "\u3000".join(res)


data = re.split(r"([。])", data)

text = []
for i in range(len(data)):
    if data[i] == '。':
        temp = data[i - 1]+data[i]
        text.append(temp)

ws = CkipWordSegmenter(level=3)(text)
pos = CkipPosTagger(level=3)(ws)
ner = CkipNerChunker(level=3)(text)

for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
    print(sentence)
    print(sentence_ws)
    print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
    for entity in sentence_ner:
        print(entity)
    print('\n')



