#coding=utf8
import pandas as pd
import numpy as np
import datetime,time
import MySQLdb
import multiprocessing
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

def layercorr(period,mfid,factor,position): #计算分层秩相关系数，拆为函数便于多进程
    grouprates = pd.concat([tradedata,position],axis=1).dropna().groupby('factor_position').mean().ix[:,'sk_yieldm']
    corr = stats.stats.spearmanr(grouprates,-grouprates.index)[0]
    command = "insert into mf_sk_layers (mf_id,periods_date,factor_name,layer_corrs,pctchange_frontend,pctchange_backend,created_at,updated_at) " \
              + "values ('%s','%s','%s',%s,%s,%s,'%s','%s')" \
              %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'),
                factor,str(round(corr,6)),
                str(round(grouprates[1],6)),str(round(grouprates[int(strategy.ix[mfid,'layer_num'])],6)),
                today,today)
    cur_aa.execute(command)
    print '%s %s corr calculated: %s' %(datetime.datetime.strftime(period,'%Y-%m-%d'),mfid,factor)

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

#读取交易日
command = "select sk_tradedate from ra_stock_yieldindic"
tradedates = pd.read_sql(command,conn_mf).ix[:,'sk_tradedate'].drop_duplicates()
print 'Tradedates loaded.'

#定义报告期
periodall = pd.Series(pd.date_range(start=datetime.datetime(1990,1,1),
                                    end=datetime.datetime.now(),freq='M'))
today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
periods = pd.Series()
for mfid in strategy.index:
    realstart = periodall[periodall.apply(lambda x: x.date())<=strategy.ix[mfid,'start_time']].iloc[-strategy.ix[mfid,'lookback_num']-1] #多-1，使得可以取到首期的上一月的因子值
    periods.ix[mfid] = pd.date_range(start=realstart,end=datetime.datetime.now(),freq='M')
periodsum = pd.Series(list(pd.date_range(start=periods.apply(min).min(),end=datetime.datetime.now(),freq='M')))

#逐时期逐策略逐因子录入因子净值
for pi in range(1,len(periodsum)):
    period = periodsum.iloc[pi] #当月收益率用当月
    periodlast = periodsum.iloc[pi-1] #因子分层应读取上月
    perioddate = max(tradedates[tradedates<=period.date()]) #此处，其实要求数据最好是全部自然日，仅有交易日数据会导致yieldm不准（包含上月的余孽）
    command = "select globalid,sk_yieldm from ra_stock_yieldindic where sk_tradedate = '%s'" \
              %(datetime.datetime.strftime(perioddate,'%Y-%m-%d'))
    tradedata = pd.read_sql(command,conn_mf).set_index('globalid').ix[:,'sk_yieldm']
    for mfid in strategy.index:
        if period in periods[mfid]:
            #计算分层秩相关系数
            command = "select factor_name,sk_code,factor_position from mf_sk_factorvalue_std where " \
                      + "periods_date = '%s' and mf_id = '%s'" \
                      %(datetime.datetime.strftime(periodlast,'%Y-%m-%d'),mfid) #因子分层应读取上月
            positions = pd.read_sql(command,conn_aa)
            p = pd.Series(index=range(0,len(factors)))
            for fi in range(0,len(factors)):
                factor = factors.iloc[fi]
                position = positions[positions.ix[:,'factor_name']==factor].set_index('sk_code').ix[:,'factor_position']
                if position.shape[0] > 1:
                    p[fi] = multiprocessing.Process(target = layercorr, args = (period,mfid,factor,position,))
                    p[fi].start()
                for fi in range(0,len(factors)):
                    try: #当p[fi]没有挂上进程，即fi非当期可用因子时会报except
                        p[fi].join()
                    except:
                        pass
            #计算因子指数 #既然还不上，就先不做了
    print 'Corr calculated: %s all.' %datetime.datetime.strftime(period,'%Y-%m-%d')
print 'All corr calculated.'

#断开数据库
cur_mf.close()
conn_mf.commit()
conn_mf.close()

cur_aa.close()
conn_aa.commit()
conn_aa.close()
