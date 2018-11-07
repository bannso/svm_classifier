import csv
import numpy as np
import random
import jieba
import jieba.posseg as psg
import pandas as pd
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.externals import joblib


def stopwordslist():
    words = set()
    filepath = "../data_qiuyi.cn/stopwords.txt"
    for line in open(filepath, 'r', encoding='utf-8').readlines():
        words.add(line.strip())
    stopwords = list(words)
    return stopwords

def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist()  # 这里加载停用词的路径
    outstr1 = ''
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr1 += word
                outstr1 += " "
    # sentence_psg = psg.cut(outstr1)
    # for x in sentence_psg:
    #     x = str(x).strip().split("/")
    #     word = x[0]
    #     part = x[1]
    #     # if part in ['a','ad','an','d','n','nz','nr','nt','ns','v','vd','vn','j','l']:
    #     if part in ['a', 'ad', 'an', 'd', 'n', 'nz', 'nr', 'nt', 'ns']:
    #         outstr += word
    #         outstr += " "
    return outstr1

def psg_sentence(sentence):
    outstr = ''
    sentence_psg = psg.cut(sentence)
    for x in sentence_psg:
        x = str(x).strip().split("/")
        word = x[0]
        part = x[1]
        if part in ['a', 'ad', 'an', 'd', 'n', 'nz', 'nr', 'nt', 'ns', 'v', 'vd', 'vn', 'j', 'l']:
            outstr += word
            outstr += " "
    return outstr

def dataSplit(percent):
    #percent:训练集在数据集中所占的比例
    # train_path = "../data/train.csv"
    # test_path = "../data/test.csv"
    train_path = "../data_qiuyi.cn/train.csv"
    test_path = "../data_qiuyi.cn/test.csv"
    f = csv.reader(open("../data_qiuyi.cn/data.csv","r",encoding="utf-8"))
    disease = []
    label = []
    title = []
    for line in f:
        # if line[3] and line[2]!= "检查":
            disease.append(line[0])
            label.append(line[1])
            title.append(line[2])
    data = list(zip(disease, label, title))
    random.shuffle(data)
    num = int(len(data) * percent)
    f1 = open(train_path,"w",newline="",encoding="utf-8")
    f2 = open(test_path,"w",newline="",encoding="utf-8")
    train = csv.writer(f1)
    test = csv.writer(f2)
    for i in range(0,len(data)):
        if i <= num:
            train.writerow(list(data[i]))
        else:
            test.writerow(list(data[i]))
    f1.close()
    f2.close()

def train(maxFeature=None,minDf=0.0,maxDf=1.0,ngram=None,classifier=svm.SVC(kernel='linear')):
    # 使用pandas读入训练和测试csv文件
    train = pd.read_csv('../data_qiuyi.cn/train.csv')
    test = pd.read_csv('../data_qiuyi.cn/test.csv')
    train.columns = ["disease","title","label"]
    test.columns = ["disease","title","label"]
# 取出句子的标签
    y_train = train['label']
    y_test = test['label']
# 将训练和测试数据都转成词list
    train_data = []
    for i in range(0, len(train['title'])):
        train_data.append(train["title"][i])
        # train_data.append(train["abstract"][i]+train['text'][i])
    test_data = []
    for i in range(0, len(test['title'])):
        test_data.append(test["title"][i])
        # test_data.append(train["abstract"][i]+test['text'][i])


    stopwords = []
    with open("../data_qiuyi.cn/stopwords.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    # 初始化TFIV对象，去停用词，加n元语言模型
    tfv = TFIV( min_df=minDf,max_df=maxDf,max_features=maxFeature, strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',ngram_range=ngram,
                  stop_words=stopwords)
    # 合并训练和测试集以便进行TFIDF向量化操作
    X_all = train_data + test_data
    tokenized_corpus = []
    # jieba.load_userdict("../data/diseaseDic.txt")
    # jieba.load_userdict("../data/symptomDic.txt")
    print("正在分词、去停...")
    for text in X_all:
        tokenized_corpus.append(" ".join(jieba.cut(seg_sentence(text))))
    len_train = len(train_data)

    # 对所有的数据进行向量化操作，耗时比较长
    print("正在抽取特征词...")
    tfv.fit(tokenized_corpus)
    feature_list = tfv.get_feature_names()
    f = open("../data_qiuyi.cn/feature_names.txt","w",encoding="utf-8")
    for x in feature_list:
        f.write(x+"\n")
    f.close()
    print("正在将文档映射为向量...")
    X_all = tfv.transform(tokenized_corpus)
    # 恢复成训练集和测试集部分
    X = X_all[:len_train]
    X_test = X_all[len_train:]


    # 定义模型结果输出函数，依次输出p值，r值，F值和准确率accuracy
    def calculate_result(actual, pred):
        m_precision = metrics.precision_score(actual, pred,average=None)
        m_recall = metrics.recall_score(actual, pred,average=None)
        accuracy = metrics.accuracy_score(pred, actual)
        print('SVM分类模型在测试集上的准确率：',accuracy)
        print('SVM分类模型在测试集上的P值：', m_precision)
        print('SVM分类模型在测试集上的R值：', m_recall)
        print('SVM分类模型在测试集上的F1值：', metrics.f1_score(actual, pred,average=None))
        return accuracy


    # 训练P_svm分类模型
    print("正在训练分类器...")
    svclf = classifier
    svclf.fit(X, y_train)
    # #模型保存
    joblib.dump(svclf,"P_svm_train_model.m")
    print("正在预测...")
    pred = svclf.predict(X_test)
    acc = calculate_result(y_test, pred)
    return acc

if __name__ == '__main__':
    # dataSplit(percent=0.8)
    train(maxFeature=21000, minDf=0, maxDf=1.0, ngram=(1, 3),classifier=svm.SVC(kernel="rbf",gamma=0.6))

