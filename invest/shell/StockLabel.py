#coding=utf8


import string
import numpy as np
from datetime import datetime
import pandas as pd


#start_date = '2007-01-05'
#end_date   = '2009-12-31'

#start_date = '2008-01-04'
#end_date   = '2010-12-31'


#start_date = '2009-01-09'
#end_date   = '2011-12-30'

#start_date = '2010-01-08'
#end_date   = '2012-12-28'

#start_date = '2011-01-07'
#end_date   = '2013-12-27'


start_date = '2012-01-06'
end_date   = '2014-12-31'


f = open('data/codes','r')
lines = f.readlines()
f.close()

codes = set()
for line in lines:
    codes.add(string.atoi(line.strip()))



f = open('./data/dates','r')
line = f.readline()
f.close()
dates = set()
vec = line.strip().split(',')
for d in vec:
        dates.add(d)


lines = open('./data/funds','r').readlines()

fundvs = {}
for line in lines:
        vec = line.split()
        code = string.atoi(vec[0].strip())
    if code not in codes:
        continue
        d    = vec[1].strip()
        date = datetime.strptime(vec[1].strip(),'%Y-%m-%d')
        if (date < datetime.strptime(start_date, '%Y-%m-%d')) or (date > datetime.strptime(end_date,'%Y-%m-%d') or (d not in dates)):
                continue
        v    = string.atof(vec[2].strip())
        vs   = fundvs.setdefault(code, [])
        vs.append(v)

#print fundvs


fundrs = {}
for code in fundvs.keys():
        vs = fundvs[code]
        rs = []
        for i in range(1, len(vs)):
                rs.append(vs[i] / vs[i-1] - 1)
        fundrs[code] = rs
#print fundrs


df = pd.read_csv('./data/index_weekly', index_col='date', parse_dates=[0])
df = df[ df.index <= datetime.strptime(end_date,'%Y-%m-%d')]
df = df[ df.index >= datetime.strptime(start_date,'%Y-%m-%d')]

indexvs = df['hs_300'].values
indexrs = []
for i in range(1, len(indexvs)):
        indexrs.append(indexvs[i] / indexvs[i-1] - 1)
hs300 = []
for i in range(3,len(indexrs)):
    n = 0
    for j in range(0,4):
        v = indexrs[i-j]
        if v >= 0:
            n = n + 1
        else:
            n = n - 1
    if n == 0:
        hs300.append(0)
    if n > 0:
        hs300.append(1)
    if n < 0:
        hs300.append(-1)



dapanindexvs = df['dapan'].values
xiaopanindexvs = df['xiaopan'].values
indexrs = []
for i in range(1, len(indexvs)):
        indexrs.append((dapanindexvs[i] / dapanindexvs[i-1] - 1) - (xiaopanindexvs[i] - xiaopanindexvs[i - 1] - 1))
daxiaopan = []
for i in range(3,len(indexrs)):
    n = 0
    for j in range(0,4):
        v = string.atof(indexrs[i-j])
        if v >= 0:
            n = n + 1
        else:
            n = n - 1
    if n == 0:
        daxiaopan.append(0)
    if n > 0:
        daxiaopan.append(1)
    if n < 0:
        daxiaopan.append(-1)



chengzhangindexvs = df['chengzhang'].values
jiazhiindexvs = df['jiazhi'].values
indexrs = []
for i in range(1, len(indexvs)):
        indexrs.append((chengzhangindexvs[i] / chengzhangindexvs[i-1] - 1) - (jiazhiindexvs[i] - jiazhiindexvs[i - 1] - 1))
chengzhangjiazhi = []
for i in range(3,len(indexrs)):
    n = 0
    for j in range(0,4):
        v = string.atof(indexrs[i-j])
        if v >= 0:
            n = n + 1
        else:
            n = n - 1
    if n == 0:
        chengzhangjiazhi.append(0)
    if n > 0:
        chengzhangjiazhi.append(1)
    if n < 0:
        chengzhangjiazhi.append(-1)


#print hs300
#print daxiaopan
#print chengzhangjiazhi




fundup = {}
funddown = {}
fundmiddle = {}
funddapan = {}
fundxiaopan = {}
fundchengzhang = {}
fundjiazhi = {}
fundupstd = {}
funddownstd = {}
fundmiddlestd = {}
funddapanstd = {}
fundxiaopanstd = {}
fundchengzhangstd = {}
fundjiazhistd = {}




rf = (0.025 / 52)

for k,v  in fundrs.items():
    code = k
    rs = []
    for j in range(3, len(v)):
        rs.append(v[j])
    
    up       = []
    down     = []
    middle   = []


    dapan    = []
    xiaopan  = []
    
    chengzhang = []
    jiazhi     = []


    for j in range(0, len(hs300)):
        p = hs300[j]
        if p == 1:
            up.append(rs[j])    
        elif p == 0:
            middle.append(rs[j])
        else:
            down.append(rs[j])


    for j in range(0, len(daxiaopan)):
        p = daxiaopan[j]
        if p >= 0:
            dapan.append(rs[j])                
        else:
            xiaopan.append(rs[j])
        

    for j in range(0, len(chengzhangjiazhi)):
        p = chengzhangjiazhi[j]
        if p >= 0 :
            chengzhang.append(rs[j])
        else:
            jiazhi.append(rs[j])

    #print rs    
    #print code , 'up' , up
    #print code , 'down', down
    #print code , 'middle', middle    
    #print code , 'dapan' , dapan
    #print code , 'xiaopan', xiaopan
    #print code , 'chengzhang', chengzhang
    #print code , 'jiazhi'    ,  jiazhi


    
    print code , 'up' , (np.mean(up) - rf) /  np.std(up)
    print code , 'down', (np.mean(down) - rf) / np.std(down)
    print code , 'middle', (np.mean(middle) - rf) / np.std(middle)    
    print code , 'dapan' , (np.mean(dapan) - rf) / np.std(dapan)
    print code , 'xiaopan', (np.mean(xiaopan) - rf) / np.std(xiaopan)
    print code , 'chengzhang', (np.mean(chengzhang) - rf) / np.std(chengzhang)
    print code , 'jiazhi'    ,  (np.mean(jiazhi) - rf) / np.std(jiazhi)
        
    fundup[code] = np.mean(up)
    funddown[code] = np.mean(down)
    fundmiddle[code] = np.mean(middle)
    funddapan[code] = np.mean(dapan)
    fundxiaopan[code] = np.mean(xiaopan)
    fundchengzhang[code] = np.mean(chengzhang)
    fundjiazhi[code] = np.mean(jiazhi)

    fundupstd[code] = np.std(up)
    funddownstd[code] = np.std(down)
    fundmiddlestd[code] = np.std(middle)
    funddapanstd[code]  = np.std(dapan)
    fundxiaopanstd[code] = np.std(xiaopan)
    fundchengzhangstd[code] = np.std(chengzhang)
    fundjiazhistd[code]     = np.std(jiazhi)        

    
#print shangzheng
#print daxiaopan
#print chengzhangjiazhi

