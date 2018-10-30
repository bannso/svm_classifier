import gensim
import numpy as np
from gensim.models import word2vec
from sklearn import metrics, svm

import xlrd
import xlwt
import xlutils.copy

import os
import sys

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV


###### 从xls文件 读取摘要 计算TF-IDF
def faction1():
    ########################## 读取摘要
    file_name_1='all_information_2.xls'
    data = xlrd.open_workbook(file_name_1)
    table = data.sheets()[0]
    nrows = table.nrows
    list_AB=[]
    for i in range(0, nrows):
        list_AB.append((table.row_values(i)[:10])[2])
    ####################################### 读取停用词
    stopwords = []
    file = open('stopwords.txt', 'r', encoding='UTF-8')
    lines = file.readlines()
    for line in lines:
        temp = line.replace('\n', "")
        stopwords.append(temp)
    file.close()

    #################
    tfv = TFIV(min_df=2, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{3,}',
               ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words=stopwords)

    tfv.fit(list_AB)
    X_all = tfv.transform(list_AB)

    file_matrix = np.matrix.tolist(X_all.todense())
    keyswords = list(tfv.vocabulary_.keys())

    f1 = open('keyswords.txt', 'w', encoding='UTF-8')
    for i in range(0, len(keyswords)):
        f1.write(str(keyswords[i]))
        f1.write('\n')
    f1.close()

    f1 = open('file_matrix_1.txt', 'w', encoding='UTF-8')
    for i in range(0, len(file_matrix)):
        for j in range(0, len(file_matrix[0])):
            f1.write(str(file_matrix[i][j]))
            f1.write(' ')
        f1.write('\n')
    f1.close()

    f1 = open('file_matrix_2.txt', 'w', encoding='UTF-8')
    for i in range(0, len(file_matrix)):
        for j in range(0, len(file_matrix[0])):
            if file_matrix[i][j] == 0:
                f1.write(str(0))
                f1.write(' ')
            else:
                f1.write(str(1))
                f1.write(' ')
        f1.write('\n')
    f1.close()
    return file_matrix

############  用word2vec计算每个词的权重
def faction2():
    keyswords = []
    file = open('keyswords.txt', 'r', encoding='UTF-8')
    lines = file.readlines()
    for line in lines:
        temp = line.replace('\n', "")
        keyswords.append(temp)
    file.close()

    vector_keyswords = []
    model = gensim.models.KeyedVectors.load_word2vec_format('PubMed-w2v.bin', binary=True)
    for i in range(0, len(keyswords)):
        try:
            vector_keyswords.append(np.matrix.tolist(model.get_vector(keyswords[i])))
        except:
            np.zeros(200)
            vector_keyswords.append(np.zeros(200))
            f1 = open('word_not_in_word2vec.txt', 'a', encoding='UTF-8')
            f1.write(str(keyswords[i]))
            f1.write('\n')
            f1.close()

    f1 = open('vector_keyswords.txt', 'w', encoding='UTF-8')
    for i in range(0, len(vector_keyswords)):
        for j in range(0, len(vector_keyswords[0])):
            f1.write(str(vector_keyswords[i][j]))
            f1.write(' ')
        f1.write('\n')
    f1.close()
    return vector_keyswords
########### 计算每篇文档的特征向量 每个词的向量 求和取均值
######## file_vector_4 摘要 绝对值求和
def faction3():
    file_matrix=faction1()
    vector_keyswords=faction2()
    num_words=0
    file_vector=[]
    for i in range(0, len(file_matrix)):
        temp_vector = np.zeros(200)
        num_words = 0

        for j in range(0, len(file_matrix[0])):

            if file_matrix[i][j] != 0:
                # temp_vector = temp_vector +vector_keyswords[j]
                temp_vector=temp_vector+np.abs(vector_keyswords[j])
                # print(vector_keyswords[j])
                # print(np.abs(vector_keyswords[j]))
                num_words=num_words+1
        if num_words!=0:
            # temp_vector=temp_vector/num_words
            temp_vector = temp_vector
        file_vector.append(temp_vector)

    f1 = open('file_vector_4.txt', 'w', encoding='UTF-8')
    for i in range(0, len(file_vector)):
        for j in range(0, len(file_vector[0])):
            f1.write(str(file_vector[i][j]))
            f1.write(' ')
        f1.write('\n')
    f1.close()
    return file_vector
########### 获得每篇文档的标签
def faction4():
    MH = {}
    a = open('PMID_include.txt', 'r', encoding='UTF-8')
    for str2 in a:
        str2 = str2.strip()
        if str2 != "":
            child = MH.get(str2)
            if not child:
                MH[str2] = int(1)
            else:
                MH[str2] = MH[str2] + 1
    a.close()

    file_name_1 = 'all_information_2.xls'
    data = xlrd.open_workbook(file_name_1)
    table = data.sheets()[0]
    nrows = table.nrows
    file_lable = []
    for i in range(0, nrows):
        str2=(table.row_values(i)[:10])[0]
        str2 = str2.strip()
        if str2 != "":
            child = MH.get(str2)
            if not child:
                file_lable.append(-1)
            else:
                file_lable.append(1)
    print(file_lable)
    print(len(file_lable))
    return file_lable
def faction5():
    file = open('file_vector_2.txt', 'r', encoding='UTF-8')
    data_arry=[]
    lines = file.readlines()
    for line in lines:
        line=line.strip()
        line=line.split(' ')
        data_arry.append(line)
    file.close()

    file_matrix=np.mat(data_arry)
    print(file_matrix.shape)
    return np.mat(file_matrix)


def faction6():

    def calculate_result(actual, pred):
        m_precision = metrics.precision_score(actual, pred,average=None)
        m_recall = metrics.recall_score(actual, pred,average=None)
        accuracy = metrics.accuracy_score(pred, actual)
        print('SVM分类模型在测试集上的准确率：',accuracy)
        print('SVM分类模型在测试集上的P值：', m_precision)
        print('SVM分类模型在测试集上的R值：', m_recall)
        print('SVM分类模型在测试集上的F1值：', metrics.f1_score(actual, pred,average=None))
        len_in=0
        len_in_2=0
        for i in range(0,len(actual)):
            if pred[i]==1:
                len_in_2 = len_in_2 + 1
                if actual[i]==1:
                    len_in=len_in+1


        print('len_in',len_in)
        print('len_in_2', len_in_2)
        return accuracy

    # 训练P_svm分类模型
    file_vector=faction5()
    file_lable=faction4()
    len_train=15563

    X_train=file_vector[:len_train]
    X_test=file_vector[len_train:]

    lable_train=file_lable[:len_train]
    lable_test=file_lable[len_train:]

    svclf = svm.SVC(kernel='rbf', C=0.3, gamma=0.4, class_weight={-1:1, 1:300})
    # svclf = svm.SVC(kernel='linear', C=0.8, class_weight={-1: 1, 1: 400})

    svclf.fit(X_train, lable_train)
    pred = svclf.predict(X_test)
    acc = calculate_result(lable_test, pred)

# faction1()
# faction2()
# faction3()
# faction4()
# faction5()
faction6()
# faction7()




