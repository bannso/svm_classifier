import csv
import time
import numpy as np
import gensim
import random
import jieba
import jieba.posseg as psg
import pandas as pd
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

def getWordVector(featureDir,embeddingDir):
    """
    将特征词映射为word embedding 向量，生成文件，并记录未能映射的词
    :param featureDir: 特征词文件路径
    :param embeddingDir: word2vec embedding路径
    :return: None
    """
    num = 0
    failList = []
    record = open("../data/record.txt","a",encoding="utf-8")
    record.write("word embedding路径："+embeddingDir+"\n" )
    with open(featureDir,"r",encoding="utf-8") as f:
        word_vector = csv.writer(open("../data/word_vector.csv","w",newline="",encoding="utf-8"))
        model = gensim.models.KeyedVectors.load_word2vec_format(embeddingDir, encoding="utf-8")
        for line in f.readlines():
            word = line.strip()
            if word in model.wv.vocab:
                vector = model.get_vector(word)
            else:
                num += 1
                failList.append(word)
                vector = np.array([0]*300)
            word_vector.writerow([word,str(vector.tolist())])
        print("字典中不存在的词向量个数：%d" %num)
    record.write("字典中不存在的此向量个数："+str(num)+'\n')
    record.write(str(failList)+"\n")
    record.close()

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
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    f = csv.reader(open("../data/data_five_labels.csv","r",encoding="utf-8"))
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
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    train.columns = ["disease","position","label","title","abstract","text"]
    test.columns = ["disease","position","label","title","abstract","text"]
    word_vec = {}
    with open("../data/word_vector.csv","r",encoding="utf-8") as f:
        for line in csv.reader(f):
            word_vec.update({line[0]:eval(line[1])})
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
    # 初始化TFIV对象，去停用词，加1-2元语言模型
    tfv = TFIV( min_df=minDf,max_df=maxDf,max_features=maxFeature, strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',ngram_range=ngram,
                  stop_words=stopwords)
    # 合并训练和测试集以便进行TFIDF向量化操作
    X_all = train_data + test_data
    tokenized_corpus = []
    # jieba.load_userdict("../data/diseaseDic.txt")
    # jieba.load_userdict("../data/symptomDic.txt")
    for text in X_all:
        tokenized_corpus.append(" ".join(jieba.cut(seg_sentence(text))))
    len_train = len(train_data)

    # 对所有的数据进行向量化操作，耗时比较长
    tfv.fit(tokenized_corpus)
    feature_list = tfv.get_feature_names()
    f = open("../data/feature_names_2.txt","w",encoding="utf-8")
    for x in feature_list:
        f.write(x+"\n")
    f.close()
    model = gensim.models.KeyedVectors.load_word2vec_format("../data/sgns.wiki.bigram-char", encoding="utf-8")
    X_all = tfv.transform(tokenized_corpus)
    all = []
    for sentence in X_all:
        sentenceWeight = np.array([0]*300)
        num = 0
        sentence = sentence.toarray().tolist()[0]
        for x in sentence:
            i = sentence.index(x)
            weight = x #使用x对词向量加权
            if x != 0:
                num += 1
                word = feature_list[i]
                if word in word_vec.keys():
                    wordMatrix = np.array(word_vec[word])
                else:
                    wordMatrix = np.array([0]*300)
                sentenceWeight = np.add(sentenceWeight,weight*wordMatrix)
        if num != 0:
            all.append((sentenceWeight/num).tolist())
        else:
            all.append([0]*300)
    with open("../data/vectors.txt","w",encoding="utf-8") as file:
        file.write(str(X_all))
    # 恢复成训练集和测试集部分
    X = np.array(all[:len_train])
    X_test = np.array(all[len_train:])


    # 定义模型结果输出函数，依次输出p值，r值，F值和准确率accuracy
    def calculate_result(actual, pred):
        m_precision = metrics.precision_score(actual, pred,average=None)
        m_recall = metrics.recall_score(actual, pred,average=None)
        accuracy = metrics.accuracy_score(pred, actual)
        with open("../data/record.txt","a",encoding="utf-8") as f:
            f.write('SVM分类模型在测试集上的准确率：'+str(accuracy)+'\n')
            f.write('SVM分类模型在测试集上的准确率：'+str(accuracy)+'\n')
            f.write('SVM分类模型在测试集上的P值：'+str(m_precision)+'\n')
            f.write('SVM分类模型在测试集上的R值：'+str(m_recall)+'\n')
            f.write('SVM分类模型在测试集上的F1值：'+str(metrics.f1_score(actual, pred,average=None))+'\n')
        return accuracy


    # 训练P_svm分类模型
    svclf = classifier
    svclf.fit(X, y_train)
    # #模型保存
    # joblib.dump(svclf,"P_svm_train_model.m")
    pred = svclf.predict(X_test)
    acc = calculate_result(y_test, pred)
    return acc

if __name__ == '__main__':
    # getWordVector("../data/feature_names_2.txt","../data/sgns.wiki.bigram-char")
    """
    SVM分类模型在测试集上的准确率： 0.8018785222291797
    SVM分类模型在测试集上的P值： [0.6961326  0.83161954 0.8195629  0.73399302]
    SVM分类模型在测试集上的R值： [0.58878505 0.75028991 0.88336685 0.74045802]
    SVM分类模型在测试集上的F1值： [0.63797468 0.78886405 0.8502696  0.73721134]
    0.8018785222291797
    """
    print("开始时间:")
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
    for dir in ["sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5",
                "sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5",
                "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5",
                "sgns.wiki.bigram",
                "sgns.wiki.char",
                "sgns.wiki.word",
                "sgns.zhihu.bigram",
                "sgns.zhihu.char"]:
        getWordVector("../data/feature_names_2.txt", "../data/"+dir)
        print(train(maxFeature=21000, minDf=0, maxDf=1.0, ngram=(1, 1)))
    print("结束时间:")
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))