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
    filepath = "../data/stopwords.txt"
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
    train_path = "../data/train_0.8.csv"
    test_path = "../data/test_0.2.csv"
    f = csv.reader(open("../data/data.csv","r",encoding="utf-8"))
    disease = []
    position = []
    label = []
    title = []
    abstract = []
    text = []
    for line in f:
        if line[3] and line[2]!= "检查":
            disease.append(line[0])
            position.append(line[1])
            label.append(line[2])
            title.append(line[3])
            abstract.append(line[4])
            text.append(line[5])
    data = list(zip(disease, position, label, title, abstract, text))
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
    train = pd.read_csv('../data/train_0.8.csv')
    test = pd.read_csv('../data/test_0.2.csv')
    train.columns = ["disease","position","label","title","abstract","text"]
    test.columns = ["disease","position","label","title","abstract","text"]
# 取出句子的标签
    y_train = train['label']
    y_test = test['label']
# 将训练和测试数据都转成词list
    train_data = []
    for i in range(0, len(train['title'])):
        train_data.append(train["title"][i]+str(train["abstract"][i])+str(train["text"][i]))
        # train_data.append(train["abstract"][i]+train['text'][i])
    test_data = []
    for i in range(0, len(test['title'])):
        test_data.append(test["title"][i]+str(test["abstract"][i])+str(test["text"][i]))
        # test_data.append(train["abstract"][i]+test['text'][i])


    stopwords = []
    with open("../data/stopwords.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    # with open("../data/diseaseDic.txt","r",encoding="utf-8") as f:
    #     lines = f.readlines();
    #     for line in lines:
    #         stopwords.append(line.strip())
    # with open("../data/symptomDic.txt","r",encoding="utf-8") as f:
    #     lines = f.readlines();
    #     for line in lines:
    #         stopwords.append(line.strip())
    # 初始化TFIV对象，去停用词，加1-2元语言模型
    tfv = TFIV( min_df=minDf,max_df=maxDf,max_features=maxFeature, strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',ngram_range=ngram,
                  stop_words=stopwords)
    # 合并训练和测试集以便进行TFIDF向量化操作
    X_all = train_data + test_data
    tokenized_corpus = []
    jieba.load_userdict("../data/diseaseDic.txt")
    jieba.load_userdict("../data/symptomDic.txt")
    print("正在分词、去停...")
    for text in X_all:
        tokenized_corpus.append(" ".join(jieba.cut(seg_sentence(text))))
    len_train = len(train_data)

    # 对所有的数据进行向量化操作，耗时比较长
    print("正在抽取特征词...")
    tfv.fit(tokenized_corpus)
    feature_list = tfv.get_feature_names()
    f = open("../data/feature_names_3.txt","w",encoding="utf-8")
    for x in feature_list:
        f.write(x+"\n")
    f.close()
    print("正在将文档映射为向量...")
    X_all = tfv.transform(tokenized_corpus)
    with open("../data/vectors.txt","w",encoding="utf-8") as file:
        file.write(str(X_all))
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
    # joblib.dump(svclf,"P_svm_train_model.m")
    print("正在预测...")
    pred = svclf.predict(X_test)
    acc = calculate_result(y_test, pred)
    return acc

if __name__ == '__main__':
    # dataSplit(percent=0.9)
    # acc = [train(5000),train(7500),train(10000),
    #        train(12500),train(15000),train(17500),
    #        train(20000),train(25000),train(30000)]
    #0.33579204092558446,0.3315091309261793,0.32606626613526857,0.3245494021771459,0.3214859318303492
    #将疾病和症状添加到停用词表
    # acc = [train(50000)]
    #0.35747427279757304,0.3529831657843079,0.35039557432633395,0.34649931592409733
    # acc = [train(20),train(400),train(600),train(800)]
    # print(acc)
    # f = csv.reader(open("../data/articles.csv","r",encoding="utf-8"))

    # print(train(maxFeature=50000, minDf=2, maxDf=1.0, ngram=(1, 2)))
    #SVM分类模型在测试集上的准确率： 0.8335629304946776
    # SVM分类模型在测试集上的P值： [0.77540107 0.84737485 0.85272379 0.77843602]
    # SVM分类模型在测试集上的R值： [0.67757009 0.8047932  0.8948578  0.77157957]
    # SVM分类模型在测试集上的F1值： [0.72319202 0.82553529 0.87328287 0.77499263]
    # print(train(maxFeature=50000, minDf=2, maxDf=1.0, ngram=(2, 2)))
    # print(train(maxFeature=50000, minDf=3, maxDf=1.0, ngram=(1, 2)))
    # print(train(maxFeature=50000, minDf=3, maxDf=1.0, ngram=(2, 2)))
    # print(train(maxFeature=20000, minDf=2, maxDf=1.0, ngram=(1, 2)))
    # print(train(maxFeature=20000, minDf=0, maxDf=1.0, ngram=(1, 2)))
    # print(train(maxFeature=20000, minDf=2, maxDf=0.9, ngram=(1, 2)))
    """
    SVM分类模型在测试集上的准确率： 0.8324358171571697
    SVM分类模型在测试集上的P值： [0.78074866 0.84555827 0.85202407 0.77665877]
    SVM分类模型在测试集上的R值： [0.68224299 0.80208736 0.8948578  0.76981797]
    SVM分类模型在测试集上的F1值： [0.72817955 0.82324936 0.87291579 0.77322324]
    0.8324358171571697
    SVM分类模型在测试集上的准确率： 0.8338134001252349
    SVM分类模型在测试集上的P值： [0.78306878 0.8477551  0.85229759 0.77928994]
    SVM分类模型在测试集上的R值： [0.69158879 0.80286046 0.89514507 0.77334116]
    SVM分类模型在测试集上的F1值： [0.73449132 0.82469724 0.87319602 0.77630416]
    0.8338134001252349
    SVM分类模型在测试集上的准确率： 0.8324358171571697
    SVM分类模型在测试集上的P值： [0.78074866 0.84555827 0.85202407 0.77665877]
    SVM分类模型在测试集上的R值： [0.68224299 0.80208736 0.8948578  0.76981797]
    SVM分类模型在测试集上的F1值： [0.72817955 0.82324936 0.87291579 0.77322324]
    0.8324358171571697
    """
    # print(train(maxFeature=20000, minDf=0, maxDf=1.0, ngram=(1, 3)))
    # 0.8340638697557922
    # print(train(maxFeature=20000, minDf=0, maxDf=1.0, ngram=(1, 2)))
    # 0.8338134001252349
    # print(train(maxFeature=10000, minDf=0, maxDf=1.0, ngram=(1, 3)))
    # 0.8309329993738259
    # print(train(maxFeature=21000, minDf=0, maxDf=1.0, ngram=(1, 3)))
    # print(train(maxFeature=21000, minDf=0, maxDf=1.0, ngram=(1, 1)))
    """
    SVM分类模型在测试集上的准确率： 0.8338134001252349
    SVM分类模型在测试集上的P值： [0.77248677 0.8527355  0.85510428 0.76873911]
    SVM分类模型在测试集上的R值： [0.68224299 0.80131426 0.89514507 0.77686436]
    SVM分类模型在测试集上的F1值： [0.72456576 0.82622559 0.87466667 0.77278037]
    0.8338134001252349
    SVM分类模型在测试集上的准确率： 0.8309329993738259
    SVM分类模型在测试集上的P值： [0.76262626 0.85099338 0.85549451 0.75909879]
    SVM分类模型在测试集上的R值： [0.70560748 0.79474295 0.89457053 0.77157957]
    SVM分类模型在测试集上的F1值： [0.73300971 0.82190686 0.87459626 0.76528829]
    0.8309329993738259
    """
    dataSplit(percent=0.8)
    train(maxFeature=21000, minDf=0, maxDf=1.0, ngram=(1, 3),classifier=svm.SVC(kernel="rbf",gamma=0.6))
    # train(maxFeature=21000, minDf=0, maxDf=1.0, ngram=(1, 1))
