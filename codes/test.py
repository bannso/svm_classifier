# # #coding=utf-8
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from codes.new_model import seg_sentence
import jieba
# import scipy.sparse.csr.csr_matrix
tfv = TFIV(strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',stop_words=None)
tfv.fit([" ".join(jieba.cut("SVM分类模型在测试集上的准确率"))," ".join(jieba.cut("本文为博主原创文章，未经博主允许不得转载"))," ".join(jieba.cut("【word2vec实例2】加载模型"))])
print(tfv.get_feature_names())
# print(tfv.transform([" ".join(jieba.cut("SVM分类模型在测试集上的准确率"))," ".join(jieba.cut("本文为博主原创文章，未经博主允许不得转载"))," ".join(jieba.cut("【word2vec实例2】加载模型"))]))
all= tfv.transform(
    [" ".join(jieba.cut("SVM分类模型在测试集上的准确率"))," ".join(jieba.cut("本文为博主原创文章，未经博主允许不得转载"))," ".join(jieba.cut("【word2vec实例2】加载模型")),"","测试"]
)
for x in all:
    x = x.toarray().tolist()[0]
    for xx in x:
        if xx!=0:
            print(x.index(xx))
# print(all[0])
# # print(all.toarray().tolist())
# # for x in all:
# #     print(x.toarray().tolist())
# # # print(str(all))
# # # # # f = open("../data/words.txt","w",encoding="utf-8")
# # # # # i=0
# # # # # for x in all:
# # # # #     print(str(all[i]))
# # # # #     f.write(str(all[i])+'\n')
# # # # #     f.write("<pass>\n")
# # # # #     i += 1
# # # # # f.close()
# # # # # # # import gensim
# # # # # # # import numpy as np
# # # # # # # model  = gensim.models.KeyedVectors.load_word2vec_format("../data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin",binary=True)
# # # # # # # vocab = model.wv.vocab
# # # # # # # print(len(model.get_vector("你").tolist()))
# # # # # # # print(len(model.get_vector("你")))
# # # # # # # word_vector = {}
# # # # # # # for word in vocab:
# # # # # # #     word_vector[word] = model[word]
# # # # # # # with open("voca.txt","w",encoding="utf-8") as f:
# # # # # # #     for k,v in word_vector.items():
# # # # # # #         f.write(k+"\t"+str(v.tolist())+"\n")
# # # # # # # model  = gensim.models.KeyedVectors.load_word2vec_format("../data/sgns.wiki.bigram-char",encoding="utf-8")
# # # # # #
# # # # with open("../data/vectors.txt") as f:
# # # #     i = 0
# # # #     j = 0
# # # #     for line in f.readlines():
# # # #         if line.strip() == "<pass>":
# # # #             i += 1
# # # #         if line.strip() == ":	:":
# # # #             j += 1
# # # #     print(i) # = 79865
# # # #     print(j)
# # # # import csv
# # # # f = csv.reader(open("../data/articles.csv","r",encoding="utf-8"))
# # # # i = 0
# # # # for line in f:
# # # #     i+=1
# # # # print(i)
# # # a = [[1,2,3],[2,3,4],[3,4,5]]
# # # print(np.array(a[:2]))
# import numpy as np
# from sklearn.decomposition import PCA
# # a = np.array([[[1,2,3,4,5],[1,2,3,23,1],[5,4,34,23,8],[4,545,0,56,0]],[[1,2,6,4,5],[10,2,53,23,1],[15,4,4,3,8],[40,5,0,6,0]],[[10,2,30,4,5],[19,2,3,3,1],[5,4,14,13,8],[14,5,10,6,0]]])
# # b = a.reshape((5,12))
# # print(b)
# # pca = PCA(n_components=2)
# # pca.fit(b)
# #
# # c = pca.fit_transform(b).tolist()
# # print(c)
# # a = np.array([1,2,3])
# # print(a)
# # left,right = 1,2
# # b = np.pad(a,(left,right),mode='constant',constant_values=(0,0))
# # print(b)\
# a = []
# a.append([1,2,34])
# a.append([1,2,4])
# a = np.array(a)
# print(a)
# pca = PCA(n_components=2)
# pca.fit(a)
# b = pca.fit_transform(a).tolist()
# print(b)
# # c = np.array([1,2,3,4])
# # all = []
# # all.append(np.pad(c, (0, 10), mode='constant', constant_values=(0, 0)).tolist())
# # print(all)
# aa = []
# a = np.array([1,2,3])
# b = np.array([1,3,3])
# # print(a)
#
# left,right = 0,20
# a = np.pad(a,(left,right-left),mode='constant',constant_values=(0,0))
# b = np.pad(b,(left,right-left),mode='constant',constant_values=(0,0))
# aa.append(a)
# aa.append(b)
# pca = PCA(n_components=1)
# pca.fit(aa)
# b = pca.fit_transform(aa).tolist()
# print(b)