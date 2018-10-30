import csv

labels = ["健康","养生","心理","男女","疾病","老人","育儿","行业"]

def read_csv(dir):
    f = csv.reader(open(dir,"r",encoding="utf-8"))
    return f

def createAll():
    file = csv.writer(open("../data_120ask/allWithChapter.csv","w",newline="",encoding="utf-8"))
    for label in labels:
        dir = "../data_120ask/" + label +"Chapters.csv"
        f = read_csv(dir)
        for line in f:
            file.writerow([line[0],line[1],line[3],line[4]])

def getDf():
    """
    统计词频
    :return:
    """
    f = read_csv("../data_120ask/allWithContent.csv")
    dict = {}
    for line in f:
        if line[0] in dict.keys():
            dict.update({line[0]:dict[line[0]]+1})
        else:
            dict.update({line[0]:0})
    print(dict)

if __name__ == '__main__':
    getDf()