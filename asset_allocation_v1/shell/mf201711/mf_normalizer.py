#coding=utf8
import pandas as pd
import numpy as np
import datetime,time
import MySQLdb
import multiprocessing
import warnings

warnings.filterwarnings("ignore")
global periods,alldays,today,strategy,factors,stdata,tradedata,validmat

#########数据库连接一段时间以后就会挂掉我要疯了啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊！！！！！！！

def is_valid(mfid,si,validtmp): #计算合法股票，拆为函数便于多进程
    stock = stocks.iloc[si]
    stockdata = tradedata[tradedata.ix[:,'globalid']==stock].set_index('sk_tradedate').ix[:,'sk_close']
    stockdata[stockdata==0] = None #stockdata为原始交易日数据，用于判断是否当日停牌
    stockshift = stockdata.ix[alldays].fillna(method='pad').shift(360) #用于判断股票发行超过1年
    stockcount = stockdata.ix[alldays].rolling(60).count() #用于判断近60日内是否超过25交易日
    for pi in range(0,len(periods[mfid])): #逐(策略的)时期判断合法性
        period = periods[mfid][pi]
        periodtime = period.date()
        if strategy.ix[mfid,'if_st'] == 1:
            try: #如果该股票没有过st记录则except
                isST = np.sum((stdata.ix[stock,'start'] <= period)&((stdata.ix[stock,'end'] >= period)|(stdata.ix[stock,'end'] == datetime.datetime.strptime('0000-00-00','%Y-%m-%d')))) #别忘了仍在st的股票没有end时间
            except:
                isST = 0
        else:
            isST = 0
        if stockshift[period] == None or stockcount[period] <= 25 or isST > 0 \
           or stockdata[max(stockdata[stockdata.index<=periodtime].index)] == None:
            validtmp[pi+si*len(periods[mfid])] = 0
        else:
            validtmp[pi+si*len(periods[mfid])] = 1

def normalizer(period,factor,factordatas): #计算标准化因子值，拆为函数便于多进程
    for mfid in strategy.index: #逐策略计算因子值
        if period in periods[mfid]: #时间需在策略的规定范围内
            factordata = factordatas[validmat[mfid][validmat[mfid].ix[:,period]==1].index].dropna() #导出当因子当期合法股票原始因子值数据
            factordata[factordata==float('inf')] = None
            factordata[factordata==float('-inf')] = None
            factordata = factordata.dropna()
            if len(factordata) > 2: #当期合法股票需多于2只，可求标准差
                stddata = (factordata - np.percentile(factordata,50)) / (factordata.std() + 1e-16) #按中位数一次标准化，+1e-16防止除0报错
                stddata[stddata>5] = 5
                stddata[stddata<-5] = -5 #去除异常值
                stddata = (stddata - stddata.mean()) / (stddata.std() + 1e-16) #按均值二次标准化
                percentnum = pd.Series(index=range(0,strategy.ix[mfid,'layer_num'])) #生成边界条件序列
                for layer in range(0,strategy.ix[mfid,'layer_num']):
                    percentnum.ix[layer] = np.percentile(stddata,100.0-100.0/strategy.ix[mfid,'layer_num']*layer)
                position = stddata.apply(lambda x: np.sum(percentnum>=x))
                try: #有时会报错，但数据已经录入了。很奇怪。
                    command = "insert into mf_sk_factorvalue_std (mf_id,periods_date,sk_code,factor_name,factor_value_std,factor_position,created_at,updated_at) values (%s);" \
                              %("),(".join(map(lambda stock: "'%s','%s','%s','%s',%s,%s,'%s','%s'" \
                                               %(mfid, datetime.datetime.strftime(period,'%Y-%m-%d'), #mf_id,periods_date
                                                 stock, factor, #sk_code,factor_name
                                                 str(round(stddata[stock],6)), #factor_value_std
                                                 str(position[stock]), #factor_position
                                                 today, today), #created_at,updated_at
                                               stddata.index))) #整理为一个命令批量插入
                    cur_aa.execute(command)
                except:
                    print mfid,period,factor

#定义计算区间 #从1990年起
alldays = pd.date_range(start=datetime.datetime(1990,1,1),
                        end=datetime.datetime.now(),freq='D')
today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')

#连接数据库
aa_base = {"host": "127.0.0.1",
            "port": 3306,
            "user": "huyang",
            "passwd": "uMXfKx03Blg3jdPmauMx",
            "db":"asset_allocation",
            "charset": "utf8"}

conn_aa  = MySQLdb.connect(**aa_base)
cur_aa   = conn_aa.cursor(MySQLdb.cursors.DictCursor)
conn_aa.autocommit(True)

mf_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"mofang",
             "charset": "utf8"}

conn_mf  = MySQLdb.connect(**mf_base)
cur_mf   = conn_mf.cursor(MySQLdb.cursors.DictCursor)
conn_mf.autocommit(True)

#读取多因子策略
command = "select * from mf_sk_paras"
strategy = pd.read_sql(command,conn_aa).set_index('mf_id')

#读取因子列表
command = "select factor_name from mf_sk_factors"
factors = pd.read_sql(command,conn_aa).ix[:,'factor_name']

#读取股票st数据
command = "select sk_code,start,end from mf_sk_stdata"
stdata = pd.read_sql(command,conn_aa)

#读取股票交易数据
command = "select globalid,sk_tradedate,sk_close from ra_stock_nav" #该表中本就只有0,3,6开头股票，因而无需做这步的筛选
tradedata = pd.read_sql(command,conn_mf)
print 'All data loaded.'

#定义报告期
periodall = pd.Series(pd.date_range(start=datetime.datetime(1990,1,1),
                                    end=datetime.datetime.now(),freq='M'))
periods = pd.Series()
for mfid in strategy.index:
    realstart = periodall[periodall.apply(lambda x: x.date())<=strategy.ix[mfid,'start_time']].iloc[-strategy.ix[mfid,'lookback_num']-1] #多-1，使得可以取到首期的上一月的因子值
    periods.ix[mfid] = pd.date_range(start=realstart,end=datetime.datetime.now(),freq='M')
periodsum = pd.date_range(start=periods.apply(min).min(),end=datetime.datetime.now(),freq='M')
#periodsum = pd.date_range(start=datetime.datetime(2015,7,31),end=datetime.datetime.now(),freq='M')

#计算股票合法性，存入临时矩阵
stocks = tradedata.ix[:,'globalid'].drop_duplicates()
validmat = pd.Series([pd.DataFrame(index=stocks,columns=periodsum)]*strategy.shape[0],index=strategy.index)
for mfid in strategy.index: #逐策略判断合法性
    validtmp = multiprocessing.Array('i',range(0,len(periods[mfid])*len(stocks))) #必须写成一长串才能在多进程间共享(内存)
    p = pd.Series(index=range(0,len(stocks)))
    for si in range(0,len(stocks)): #逐股票筛出交易数据
        p[si] = multiprocessing.Process(target = is_valid, args = (mfid,si,validtmp,))
        p[si].start()
        print 'Validation calculated: %s %s' %(mfid,stocks.iloc[si])
        #time.sleep(0.02)
    for si in range(0,len(stocks)):
        p[si].join()
    for si in range(0,len(stocks)):
        for pi in range(0,len(periods[mfid])):
            validmat[mfid].ix[stocks.iloc[si],periods[mfid][pi]] = validtmp[pi+si*len(periods[mfid])]
    print 'Validation calculated: %s all.' %mfid
print 'All validation calculated.'

#计算标准化因子值
for period in periodsum: #逐所有策略时期的并集进行数据读取，是最少计算量方案
    command = "select sk_code,factor_value,factor_name from mf_sk_factorvalue where periods_date = '%s'" \
              %(datetime.datetime.strftime(period,'%Y-%m-%d'))
    perioddata = pd.read_sql(command,conn_aa) #数据库不能放到多进程中读取，否则会报错
    p = pd.Series(index=range(0,len(factors)))
    for fi in range(0,len(factors)): #逐时期和因子读取原始因子值数据
        factor = factors.iloc[fi]
        factordatas = perioddata[perioddata.ix[:,'factor_name']==factor]
        if factordatas.shape[0] > 2: #当期股票需多于2只
            factordatas = factordatas.set_index('sk_code').ix[:,'factor_value']
            p[fi] = multiprocessing.Process(target = normalizer, args = (period,factor,factordatas,))
            p[fi].start()
            print period,'data calculated: %s factor %s' %(str(factordatas.shape[0]),factor)
            time.sleep(0.3) #不能设置太低否则数据库会断线
    for fi in range(0,len(factors)):
        try: #当p[fi]没有挂上进程，即fi非当期可用因子时会报except
            p[fi].join()
        except:
            pass
    print 'Stddata calculated: %s all.' %datetime.datetime.strftime(period,'%Y-%m-%d')
print 'All stddata calculated.'

#断开数据库
cur_mf.close()
conn_mf.commit()
conn_mf.close()

cur_aa.close()
conn_aa.commit()
conn_aa.close()
