#coding=utf8
import pandas as pd
import datetime,time
import os,sys
import math
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shutil

locpath = sys.path[0]

try:
    if_ST = int(sys.argv[1])
except:
    if_ST = False

rpath = u'/home/huyang/MultiFactors201710/originalData/'
STstock = pd.read_csv('/home/huyang/MultiFactors201710/STstock.csv',index_col='code',parse_dates=['start','end']).fillna(datetime.datetime.now())

#生成至今为止的时间节点序列
d90 = []
for year in range(2004,datetime.datetime.now().year):
    for month in range(0,12):
        d90 += [datetime.datetime(year,month+1,1)]
for month in range(0,7):
    d90 += [datetime.datetime(datetime.datetime.now().year,month+1,1)]
d90 += [datetime.datetime(2017,8,1),datetime.datetime(2017,9,1)]#,datetime.datetime(2017,10,1)]
d90 = map(lambda x: x-datetime.timedelta(days=1),d90)

def daystr(datetime):
    return str(datetime)[0:4]+str(datetime)[5:7]+str(datetime)[8:10]

#清除分隔符函数
def separatorRemover(seri):
    return seri.apply(lambda x: float(str(x).replace(',','').replace(' ','')))


#检索所有有意义的股票代码，便于后面按各股票进行数值计算
codetrue = sorted(map(lambda x: int(x.replace('.csv','')), os.listdir(rpath+'TQ_SK_YIELDINDIC_bycode/')))


#有效股票统计（即发行至少超过1年，停牌1个月以上的复牌至少超过1个月（即代表日前60自然日内至少有25个非停牌的交易日））
tradedates = pd.Series(map(lambda x: datetime.datetime.strptime(str(x)[0:8],'%Y%m%d'), os.listdir(rpath+'TQ_SK_YIELDINDIC/')))
tradedates = tradedates[tradedates<datetime.datetime(2017,9,1)].dropna().sort_values()
representDates = []
for date in d90[1:len(d90)]:
    datedelta = (tradedates - date).apply(lambda x: abs(x.days))
    representDates += [tradedates[datedelta.idxmin()]]
    print date

validcodes = []
for i in d90[1:len(d90)]:
    validcodes += [[]]
for code in codetrue:
    if len(str(code))<6 or str(code)[0]=='3' or str(code)[0]=='6':#如果false，这是非主板/新三板股票
        data = pd.read_csv(rpath+'TQ_SK_YIELDINDIC_bycode/'+str(code)+'.csv')
        codedates = data.ix[:,'TURNRATE']
        codedates[codedates==0] = float('Nan')
        codedates.index = data.ix[:,'TRADEDATE'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d'))
        #if codedates.index[len(codedates)-1] == tradedates[len(tradedates)-1]:#如果false，这是已退市股票 #理论上来讲这个条件应该放松掉
        coderolling = codedates.rolling(60).count()
        if codedates.index[0] == tradedates.iloc[0]:#如果样本期一开始股票就已上市，则认为它不是新股
            for i in range(0,len(representDates)):
                try:#当股票在representDates[i]上没有数据时（不是停牌，是没有数据），会报except，下同
                    if coderolling[representDates[i]] > 25 and pd.notnull(codedates[representDates[i]]) :#把当期正在停牌的股票也去除
                        if if_ST:
                            try:
                                isST = np.sum((STstock.ix[code,'start'] <= representDates[i])&(STstock.ix[code,'end'] >= representDates[i]))
                                if isST > 0:
                                    pass
                                else:
                                    validcodes[i] += [code]
                            except:
                                validcodes[i] += [code]
                        else:
                            validcodes[i] += [code] #以上，可选是否加入对(*)ST的过滤
                except:
                    pass
        else:
            codestart = codedates.index[0] + datetime.timedelta(360)
            for i in range(0,len(representDates)):
                if representDates[i] > codestart:#如果股票样本期开始之后才上市，则其上市一年后才会纳入计算
                    try:
                        if coderolling[representDates[i]] > 25 and pd.notnull(codedates[representDates[i]]) :#把当期正在停牌的股票也去除
                            if if_ST:
                                try:
                                    isST = np.sum((STstock.ix[code,'start'] <= representDates[i])&(STstock.ix[code,'end'] >= representDates[i]))
                                    if isST > 0:
                                        pass
                                    else:
                                        validcodes[i] += [code]
                                except:
                                    validcodes[i] += [code]
                            else:
                                validcodes[i] += [code] #以上，可选是否加入对(*)ST的过滤
                    except:
                        pass
        print code

for i in range(0,len(validcodes)):
    validcode = pd.Series(validcodes[i])
    validcode.to_csv(locpath+u'/validcode/'+daystr(d90[i+1])+'.csv',header=True)
    print d90[i+1]

shutil.copy('%s/validcode/20170831.csv' %locpath,'%s/validcode/20170930.csv' %locpath)
