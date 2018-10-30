from codes.model import *

#{'症状': 24697, '病因': 8472, '检查': 4191, '鉴别': 1105, '治疗': 25620, '饮食护理': 10522, '预防': 7246, '并发症': 2200}
# train_2(labelList=["症状","病因","治疗","饮食护理"])
# train_2(labelList=["症状","治疗","饮食护理"])
scores = train_2(labelList=[],maxLen=20000,kFold=10)
print(scores)

# scores = train_3(labelList=["症状","治疗"],maxLen=10000)
# print(scores)
