import requests
from bs4 import BeautifulSoup
import csv
import random
import time

def sendRequest(url):
    Headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'}
    requests.adapters.DEFAULT_RETRIES = 5  # 增加重连次数
    s = requests.session()
    s.keep_alive = False
    html = s.get(url, headers=Headers, timeout=30)
    html.encoding = "utf-8"
    page = html.text
    soup = BeautifulSoup(page, features='html.parser')
    return soup

def getContent():
    pass

def getDiseasePage():
    f = csv.writer(open("../data/diseaseUrl.csv","a",newline="",encoding="utf-8"))
    root = "http://jb.9939.com/jbzz/"
    positions = ["头部", "颈部", "胸部", "腹部", "腰部", "男性生殖",
                 "女性生殖", "全身", "上肢", "下肢", "心理"]
    urls = [root+"toubu_t1/",root+"jingbu_t1/",root+"xiongbu_t1/",
            root+"fubu_t1/",root+"yaobu_t1/",root+"nanxingshengzhi_t1/",
            root+"nvxingshengzhi_t1/",root+"quanshen_t1/",root+"shangzhi_t1/",
            root+"xiazhi_t1/",root+"xinli_t1/"]
    for url in urls:
        print("目前访问的URL根路径："+url)
        position = positions[urls.index(url)]
        soup = sendRequest(url)
        pageNums = soup.find(attrs={"class":"paget paint"}).findAll("a")[-2].string     #获取页数
        print("当前路径下有%s页"%pageNums)
        for i in range(1,int(pageNums)+1):
            print("正在获取第%d页"%i)
            diseaseListUrl = url + "?page="+str(i)
            diseaseListPage = sendRequest(diseaseListUrl)
            for x in diseaseListPage.findAll(attrs={"class":"doc_anwer disline"}):
                diseaseWithUrl = [x.find(attrs={"class":"subtit dyh"}).find(attrs={"class":"cation fl"}).find(attrs={"class":"mr11 fl"}).find("a").get("title"),x.find(attrs={"class":"subtit dyh"}).find(attrs={"class":"cation fl"}).find(attrs={"class":"mr11 fl"}).find("a").get("href"),position]
                f.writerow(diseaseWithUrl)

def getArticlesUrl():
    f1 = csv.reader(open("../data/diseaseUrl.csv", "r", encoding="utf-8"))
    f2 = csv.writer(open("../data/articleUrl.csv","w",newline="",encoding="utf-8"))
    for line in f1:
        print("正在获取%s下文章的分类信息"%(line[0]))
        disease = line[0]
        url = line[1]
        position = line[2]
        labels = ["症状","病因","检查","鉴别","治疗","饮食护理","预防","并发症"]#症状+病因+鉴别  饮食护理+预防  治疗  并发症  检查
        for i in range(1,9):
            label = labels[i-1]
            newUrl = url + "article_list_t" + str(i) + ".shtml"
            print(newUrl)
            soup = sendRequest(newUrl)
            try:
                #有多页文章
                pageNums = soup.find(attrs={"class":"paget paint"}).findAll("a")[-4].string
                for j in range(1,int(pageNums)+1):
                    soup = sendRequest(newUrl + "?page="+str(j))
                    links = soup.find(attrs={"class": "newlis fitmou"}).findAll("a")
                    for link in links:
                        title = link.get("title")
                        articleUrl = "http://jb.9939.com" + link.get("href")
                        f2.writerow([disease,position,label,title,articleUrl])
            except:
                try:
                    #只有一页文章
                    links = soup.find(attrs={"class":"newlis fitmou"}).findAll("a")
                    for link in links:
                        title = link.get("title")
                        articleUrl = "http://jb.9939.com" + link.get("href")
                        f2.writerow([disease, position, label, title, articleUrl])
                except:
                    #没有相关文章
                    f2.writerow([disease, position, label, "None", "None"])

def getTreatmentMethod():
    f1 = csv.reader(open("../data/diseaseUrl.csv","r",encoding="utf-8"))
    f2 = csv.writer(open("../data/diseaseTreatment.csv","w",newline="",encoding="utf-8"))
    for line in f1:
        print("正在获取%s的治疗方式"%(line[0]))
        disease = line[0]
        url = line[1]
        position = line[2]
        soup = sendRequest(url+"zl/")
        passages = soup.find(attrs={"class":"tost nickn bshare prevp spread graco"}).findAll("p")
        treatmentMethod = passages[0].get_text()
        treatmentContent = passages[1].string
        # for passage in passages:
        #     if passage.get("class") == None:
        #         treatmentMethod = passage.string.split(" ")
        # treatmentContent = soup.find(attrs={"class":"tost nickn bshare prevp spread graco"}).find(attrs={"class":"spea"}).string
        f2.writerow([disease,position,treatmentMethod,treatmentContent])
        # break

def getArticles():
    finished = []
    with open("../data/logs.txt","r",encoding="utf-8") as f:
        for line in f.readlines():
            finished.append(line.strip())
    f1 = csv.reader(open("../data/articleUrl.csv", "r", encoding="utf-8"))
    f2 = csv.writer(open("../data/articles.csv", "w", newline="", encoding="utf-8"))
    thisDisease = ""
    thisLabel = ""
    for line in f1:
        disease = line[0]
        position = line[1]
        label = line[2]
        title = line[3]
        articleUrl = line[4]
        if thisDisease != disease or thisLabel != label:
            thisDisease = disease
            thisLabel = label
            print("正在获取%s疾病%s标签下文章的内容..."%(thisDisease,label))
        if articleUrl not in finished:
            soup = sendRequest(articleUrl)
            try:
                abstract = soup.find(attrs={"class": "art_wra camat"}).find(attrs={"class": "art_l"}).find(
                    attrs={"class": "a_const abstra"}).find("p").get_text().strip()
            except:
                abstract = ""
            try:
                body = " ".join(x.get_text().strip() for x in
                                soup.find(attrs={"class": "art_wra camat"}).find(attrs={"class": "art_l"}).find(
                                    attrs={"class": "a_bod"}).findAll("p"))
            except:
                body = ""
            try:
                text = body + " ".join(x.get_text().strip() for x in
                                       soup.find(attrs={"class": "art_wra camat"}).find(attrs={"class": "art_l"}).find(
                                           attrs={"class": "a_adv clearfix"}).findAll("p"))
            except:
                text = body + ""
            f2.writerow([disease, position, label, title, abstract, text])
            with open("../data/logs.txt","w",encoding="utf-8") as f:
                f.write(articleUrl+"\n")

def teatmentMethodProcess():
    f = csv.reader(open("../data/diseaseTreatment.csv","r",encoding="utf-8"))
    treatmentMethods = set()
    i = 0
    for line in f:
        # if not i:
        #     text = line[2].strip().split(" ")
        #     print(text)
        #     for x in text:
        #         if ".对症治疗"in x:
        #             print(line[0])
        #             i = 1
        text = line[2].strip().split(" ")
        for x in text:
            # if 0 < len(x) < 10 and line[0] != "胆囊癌" and line[0] != "恶性黑色素瘤":
            if len(x) >0:
                treatmentMethods.add(x)
        i += 1
    print(treatmentMethods)
    print(len(treatmentMethods))

def propress():
    f1 = csv.reader(open("../data/articles.csv","r",encoding="utf-8"))
    f2 = csv.writer(open("../data/data_part1.csv","w",newline="",encoding="utf-8"))
    i = 0
    for line in f1:
        if len(line[4]) > 0 or len(line[5]) > 0:
            i += 1
            text = line[5]
            string1 = "/*重点阅读*/#Preadblock{height:230px;position:relative;}#Pread{padding:10px0px6px10px;width:546px;color:#404040;border:#ED4A86solid1px;text-align:left;position:absolute;}#Preadtd{line-height:24px;}#Pread.una{border-left:1pxsolid#ccc;border-right:1pxsolid#ccc}#Pread.unaa{text-decoration:underline}#Pread.unaa:hover{text-decoration:none}#Preada{color:#000;text-decoration:none}#Preadimg{margin-bottom:5px}#Pread.nw{border:#06csolid1px;background:#e8eff7;color:#000}#Pread.nwtd{background-position:bottom;background-repeat:repeat-x;}#Pread.nwspan{margin:020px010px}#Pread.nwa{color:#000}.zdyd_r{color:#ED4A86;font-weight:800;background-color:#fff;font-size:12px;padding:05px;position:absolute;top:-12px;margin-left:10px;}.jdht_r{color:#ED4A86;font-weight:800;background-color:#fff;font-size:12px;padding:05px;position:absolute;top:-12px;margin-left:280px;}"
            string2 = """祛疤亲身体验
                						                	两次激光祛疤，疤痕真的掉了。						                	祛疤啦祛疤啦~~~						                	做了冷冻祛疤，疤是没了，但痕还在啊~"""
            string3 = "（责任编辑：jbwq）"
            if string1 in line[5]:
                text = "".join(text.split(string1))
            if string2 in line[5]:
                text = "".join(text.split(string2))
            if string3 in line[5]:
                text = "".join(text.split(string3))
            f2.writerow([line[0],line[1],line[2],line[3],line[4],text])
    print("文章数量%d"%(i))
def createDataWithContent(percent):
    """
    创建训练集和测试集，包含全文和摘要，[disease,position,label,title,abstract,text]
    :param percent: 训练集在数据集所占比例
    :return: None
    """
    f1 = csv.writer(open("../data/trainWithContent.csv","w",newline="",encoding="utf-8"))
    f2 = csv.writer(open("../data/testWithContent.csv","w",newline="",encoding="utf-8"))
    with open("../data/data.csv","r",encoding="utf-8") as f:
        disease = []
        position = []
        label = []
        title = []
        abstract = []
        text = []
        for line in csv.reader(f):
            if line[2] in ["症状","病因","治疗"]:
                disease.append(line[0])
                position.append(line[1])
                label.append(line[2])
                title.append(line[3])
                abstract.append(line[4])
                text.append(line[5])
        data = list(zip(disease, position, label, title, abstract, text))
        random.shuffle(data)
        num = int(len(data) * percent)
        for i in range(0, len(data)):
            if i <= num:
                f1.writerow(list(data[i]))
            else:
                f2.writerow(list(data[i]))

def getLabelFreq(dir,index):
    """
    获取文件中每个标签的数据数量
    :param dir: 文件路径
    :param index: 标签所在列的索引
    :return: dict格式，{label:num}
    """
    dict = {}
    with open(dir,"r",encoding="utf-8") as f:
        for line in csv.reader(f):
            if line[index] in dict.keys():
                dict[line[index]] += 1
            else:
                dict[line[index]] = 1
    return dict
if __name__ == '__main__':
    # createDataWithContent(0.8)
    print(getLabelFreq("../data/data.csv",2))
    # time1 = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # getArticlesUrl()
    # time2 = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # print("获取文章URL完成时间：%s" % (time1))
    # print("获取文章内容完成时间：%s" %(time2))
    #获取文章URL开始时间：2018-10-07 21:49:37 获取文章URL完成时间：2018-10-08 11:45:09
