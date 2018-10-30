import requests
from bs4 import BeautifulSoup
import csv
import gensim
import time


def sendRequest(url):
    """

    :param url:需要爬取的链接
    :return: 该链接的Beautifulsoup对象
    """
    Headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'}
    requests.adapters.DEFAULT_RETRIES = 5  # 增加重连次数
    s = requests.session()
    s.keep_alive = False
    html = s.get(url, headers=Headers, timeout=30)
    html.encoding = "utf-8"
    page = html.text
    soup = BeautifulSoup(page, features='html.parser')
    return soup

def getContent(url):
    """
    获取文章内容
    :param url:文章的链接
    :return: 文章内容，string格式
    """
    soup = sendRequest(url)
    passages = soup.find(attrs={"class":"articleCont"}).findAll("p")
    content = "".join([passage.get_text().strip() for passage in passages ])
    return content

def getArticleList(url):
    """
    获取链接下的全部文章标题和url
    :param url: 需要爬取的链接
    :return: dict格式，{"url":"title"}
    """
    soup = sendRequest(url)
    articleDivs = soup.find(attrs={"class":"article-list"}).findAll(attrs={"class":"part clears"})
    articleUrls = {"http:"+div.find(attrs={"class":"title"}).get("href"):div.find(attrs={"class":"title"}).string for div in articleDivs}
    return articleUrls

def createLabelDic():
    """
    获取文章层级
    :return: dict格式,{label_1:{label_2:url}}
    """
    labelDic = {}
    with open("../data_120ask/labelsUrl.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
        label_1 = ""
        for line in lines:
            if len(line.strip())<10:
                label_1 = line.strip()
                labelDic.update({label_1:{}})
            else:
                labelDic[label_1].update({line.strip().split("    ")[1]:line.strip().split("    ")[0]})
    return labelDic

def getTrainData():
    """
    获取全部文章的链接
    :return: None
    """
    f = csv.writer(open("../data_120ask/data_part4.csv", "a", newline="", encoding="utf-8"))
    lableDic = createLabelDic()
    for k1,v1 in lableDic.items():
        label_1 = k1
        print(label_1)
        for k2,v2 in v1.items():
            print(k2)
            label_2 = k2
            label_2_url = v2
            i = 1
            while i<=400:
                print(i)
                listUrl = label_2_url +"/" + str(i)
                articleUrls = getArticleList(listUrl)
                if articleUrls:
                    for url, title in articleUrls.items():
                            #暂时不获取全文
                        # content = getContent(url)
                        f.writerow([i,label_1, label_2, title, url])
                    i += 1
                else:
                    break

def deduplication():
    """
    对初步获取到的文章url去重，以url唯一为依据，生成allUrl文件，格式[level1,level2,title,url]
    :return: None
    """
    f = csv.writer(open("../data_120ask/allUrl.csv","w",newline="",encoding="utf-8"))
    urlList = []
    for i in range(1,5):
        print(i)
        dir = "../data_120ask/data_part" + str(i) +".csv"
        fo = csv.reader(open(dir,"r",encoding="utf-8"))
        for line in fo:
            url = line[4]
            if url not in urlList:
                urlList.append(url)
                f.writerow([line[1],line[2],line[3],line[4]])

def getAllContent(dir):
    """
    获取全部文章内容，成功获取的存入all.csv，失败的存入err.csv
    :param dir: 文章url源路径
    :return: None
    """
    fo = csv.reader(open(dir,"r",encoding="utf-8"))
    f = csv.writer(open("../data_120ask/all.csv","w",newline="",encoding="utf-8"))
    fe = csv.writer(open("../data_120ask/err.csv","w",newline="",encoding="utf-8"))
    i = ""
    for line in fo:
        try:
            if i != line[1]:
                i = line[1]
                print(i)
            print(line[3])
            content = getContent(line[3])
            f.writerow([line[0],line[1],line[3],line[2],content])
        except:
            fe.writerow([line[0],line[1],line[2],line[3]])

def generateVectors(wordDir,vectorDir):
    """
    将特征词映射为词向量，生成词向量文件
    :param wordDir:词表路径
    :param vectorDir: word embedding路径
    :return: None
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(vectorDir, encoding="utf-8")
    voca = model.wv.vocab
    word2vec =  csv.writer(open("../data/wordToVec.csv","w",newline="",encoding="utf-8"))
    with open(wordDir,"r",encoding="utf-8") as f:
        for line in f.readlines():
            word = line.strip()
            if word in voca:
                vec = model.get_vector(word).tolist()
                word2vec.writerow([word,str(vec)])
            else:
                word2vec.writerow([word, str([0]*300)])

if __name__ == '__main__':
    getAllContent("../data_120ask/allUrl.csv")