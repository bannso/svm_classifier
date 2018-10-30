import csv
import time
import jieba
import random
import jieba.posseg as psg
import pandas as pd
from sklearn import metrics, svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.externals import joblib

def messUp():
    disease,position,label,title,abstract,content = [],[],[],[],[],[]
    for line in csv.reader(open("../data/articles.csv","r",encoding="utf-8")):
        disease.append(line[0])
        position.append(line[1])
        label.append(line[2])
        title.append(line[3])
        abstract.append(line[4])
        content.append(line[5])
    data = list(zip(disease,position,label,title,abstract,content))
    random.shuffle(data)
    f_csv = csv.writer(open("../data/data_messed.csv","w",newline="",encoding="utf-8"))
    for x in data:
        f_csv.writerow(list(x))


def getDfDic(fileDir):
     dfDic = {}
     with open(fileDir,"r",encoding="utf-8") as f:
         for line in csv.reader(f):
             if line[2] in dfDic.keys():
                 dfDic[line[2]] += 1
             else:
                 dfDic[line[2]] = 0
     return dfDic

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
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

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

def getTime(string):
    print(string)
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))

def train_1():
    # 使用pandas读入数据集
    all_data = pd.read_csv('../data/articles.csv')
    all_data.columns = ["disease","position","label","title","abstract","text"]
# 取出句子的标签
    y_data = all_data['label']
    print("初始标签长度:%d" %len(y_data))
# 将训练和测试数据都转成词list
    x_data = []
    for i in range(0, len(all_data['title'])):
        x_data.append(all_data["title"][i])
    stopwords = []
    #创建停用词表
    with open("../data/stopwords.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    # 初始化TFIV对象，去停用词，加1-2元语言模型
    tfv = TFIV( min_df=0,max_df=1,max_features=30000, strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',
                  stop_words=stopwords,ngram_range=(1,3))
    x_all = []
    y_all = []
    jieba.load_userdict("../data/diseaseDic.txt")
    jieba.load_userdict("../data/symptomDic.txt")
    print("正在分词、去停...")
    i= 0
    for text in x_data:
        i+=1
        print("第%d篇文章..."%i)
        try:
            x_all.append(" ".join(jieba.cut(seg_sentence(text))))
            y_all.append(y_data[x_data.index(text)])
        except:
            pass
    print("分词去停后标签长度:%d" %len(y_all))
    print("正在抽取特征词...")
    tfv.fit(x_all)
    feature_list = tfv.get_feature_names()
    #将特征词表写入txt
    f = open("../data/feature_names_3.txt","w",encoding="utf-8")
    for x in feature_list:
        f.write(x+"\n")
    f.close()
    print("正在将文档映射为向量...")
    X_all = tfv.transform(x_all)
    clf = svm.SVC(kernel='linear',C=1)
    print("训练...")
    scores = cross_val_score(clf,X_all,y_all,cv=5)
    print(scores)

def train_2(labelList=[],maxLen=None,kFold=0):
    # 使用pandas读入数据集
    all_data = pd.read_csv('../data/data_five_labels.csv')
    all_data.columns = ["disease","position","label","title","abstract","text"]
# 将训练和测试数据都转成词list
    x_data = []
    y_data = []
    #{'症状': 24697, '病因': 8472, '检查': 4191, '鉴别': 1105, '治疗': 25620, '饮食护理': 10522, '预防': 7246, '并发症': 2200}
    for i in range(0, len(all_data['title'])):
        if all_data["label"][i] not in labelList:#剔除数据量不足的标签
            x_data.append(all_data["title"][i])
            y_data.append(all_data["label"][i])
    stopwords = []
    #创建停用词表
    with open("../data/stopwords.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    # 初始化TFIV对象，去停用词，加1-2元语言模型
    tfv = TFIV( min_df=0,max_df=1,max_features=maxLen, strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',
                  stop_words=stopwords,ngram_range=(1,3))
    x_all = []
    y_all = []
    jieba.load_userdict("../data/diseaseDic.txt")
    jieba.load_userdict("../data/symptomDic.txt")
    print("正在分词、去停...")
    i= 0
    for text in x_data:
        i+=1
        print("第%d篇文章..."%i)
        try:
            x_all.append(" ".join(jieba.cut(seg_sentence(text))))
            y_all.append(y_data[x_data.index(text)])
        except:
            pass
    print("初始标签长度:%d" % len(y_data))
    print("分词去停后标签长度:%d" %len(y_all))
    print("正在抽取特征词...")
    tfv.fit(x_all)
    feature_list = tfv.get_feature_names()
    #将特征词表写入txt
    f = open("../data/feature_names_3.txt","w",encoding="utf-8")
    for x in feature_list:
        f.write(x+"\n")
    f.close()
    print("正在将文档映射为向量...")
    X_all = tfv.transform(x_all)
    clf = svm.SVC(kernel='linear',C=1)
    print("训练...")
    scores = cross_val_score(clf,X_all,y_all,cv=kFold)
    return scores

def train_3(labelList=[],maxLen=None):
    # 使用pandas读入数据集
    all_data = pd.read_csv('../data/data_five_labels.csv')
    all_data.columns = ["disease","position","label","title","abstract","text"]
# 将训练和测试数据都转成词list
    x_data = []
    y_data = []
    #{'症状': 24697, '病因': 8472, '检查': 4191, '鉴别': 1105, '治疗': 25620, '饮食护理': 10522, '预防': 7246, '并发症': 2200}
    for i in range(0, len(all_data['title'])):
        if all_data["label"][i] in labelList:#剔除数据量不足的标签
            x_data.append(all_data["title"][i])
            y_data.append(all_data["label"][i])
    stopwords = []
    #创建停用词表
    with open("../data/stopwords.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    # 初始化TFIV对象，去停用词，加1-2元语言模型
    tfv = TFIV( min_df=0,max_df=1,max_features=maxLen, strip_accents='unicode', analyzer='word', token_pattern=r'\b[\u4e00-\u9fa5]\w+\b',
                  stop_words=stopwords,ngram_range=(1,3))
    x_all = []
    y_all = []
    jieba.load_userdict("../data/diseaseDic.txt")
    jieba.load_userdict("../data/symptomDic.txt")
    print("正在分词、去停...")
    i= 0
    for text in x_data:
        i+=1
        print("第%d篇文章..."%i)
        try:
            x_all.append(" ".join(jieba.cut(seg_sentence(text))))
            y_all.append(y_data[x_data.index(text)])
        except:
            pass
    print("初始标签长度:%d" % len(y_data))
    print("分词去停后标签长度:%d" %len(y_all))
    print("正在抽取特征词...")
    tfv.fit(x_all)
    feature_list = tfv.get_feature_names()
    #将特征词表写入txt
    f = open("../data/feature_names_3.txt","w",encoding="utf-8")
    for x in feature_list:
        f.write(x+"\n")
    f.close()
    print("正在将文档映射为向量...")
    X_all = tfv.transform(x_all)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    print("训练...")
    clf = svm.SVC(kernel='linear', C=1).fit(X_train,y_train)
    def calculate_result(actual, pred):
        m_precision = metrics.precision_score(actual, pred,average=None)
        m_recall = metrics.recall_score(actual, pred,average=None)
        accuracy = metrics.accuracy_score(pred, actual)
        print('SVM分类模型在测试集上的准确率：',accuracy)
        print('SVM分类模型在测试集上的P值：', m_precision)
        print('SVM分类模型在测试集上的R值：', m_recall)
        print('SVM分类模型在测试集上的F1值：', metrics.f1_score(actual, pred,average=None))
        return accuracy

    pred = clf.predict(X_test)
    acc = calculate_result(y_test, pred)
    return acc

if __name__ == '__main__':
    messUp()

