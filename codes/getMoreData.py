#扩充数据集 从http://jibing.qiuyi.cn/爬取数据
import requests
from bs4 import BeautifulSoup
import csv
import random
import logging
import time

log = logging.getLogger(__name__)

def sendRequest(url):
    Headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.75 Safari/537.36'}
    requests.adapters.DEFAULT_RETRIES = 5  # 增加重连次数
    s = requests.session()
    s.keep_alive = False
    html = s.get(url, headers=Headers, timeout=400)
    html.encoding = "utf-8"
    page = html.text
    soup = BeautifulSoup(page, features='html.parser')
    return soup
def getTitles():
    file = csv.writer(open("../data_qiuyi.cn/more_data.csv","a",newline="",encoding="utf-8"))
    for i in range(10,1710):
        url = "http://ci.qiuyi.cn/html/disease/"+str(i)+"/index.html"
        soup = sendRequest(url)
        try:
            print(i)
            divs = soup.findAll(attrs={"class":"r_2 bor"})
            for div in divs:
                try:
                    # print(div)
                    label = div.find("h3").string[-2:]
                    # print(label)
                    if label in ["症状","治疗","诊断","病因"]:
                        disease = div.find("h3").string[:-2]
                        links = [x.string for x in div.findAll("a")]
                        href = [x.get("href") for x in div.findAll("a")]
                        print(links)
                        for x in links:
                            file.writerow([disease,x,label,href[links.index(x)]])
                except:
                    pass
        except:
            pass

def getContents1():
    f1 = csv.reader(open("../data_qiuyi.cn/more_data.csv","r",encoding="utf-8"))
    f2 = csv.writer(open("../data_qiuyi.cn/contents.csv","a",newline="",encoding="utf-8"))
    f3 = csv.writer(open("../data_qiuyi.cn/errs.csv","w",newline="",encoding="utf-8"))
    i = 0
    for line in f1:
        url = line[3]
        print(url)
        try:
            soup = sendRequest(url)
            ps = [x.get_text() for x in soup.find(attrs={"class": "article_list1 article_details"}).findAll("p")]
            content = "".join(ps[2:-1])
            line = line + [content]
            f2.writerow(line)
            i += 1
            print(i)
        except:
            f3.writerow(line)

def getContents2():
    f1 = csv.reader(open("../data_qiuyi.cn/errs_2.csv","r",encoding="utf-8"))
    f2 = csv.writer(open("../data_qiuyi.cn/contents.csv","a",newline="",encoding="utf-8"))
    f3 = csv.writer(open("../data_qiuyi.cn/errs_3.csv","w",newline="",encoding="utf-8"))
    i = 0
    for line in f1:
        url = line[3]
        print(url)
        try:
            i += 1
            print(i)
            soup = sendRequest(url)
            ps = [x.get_text() for x in soup.find(attrs={"class": "info_area"}).findAll("p")]
            content = "".join(ps)
            line = line + [content]
            f2.writerow(line)
        except:
            f3.writerow(line)


if __name__ == '__main__':
    getContents2()

