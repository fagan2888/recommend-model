#coding=utf8
import pandas as pd
import numpy as np
import datetime,time
import MySQLdb
import multiprocessing
import os
import warnings
warnings.filterwarnings("ignore")

def fundcsvi(i): #导出位次因子股票池持仓，拆为函数便于多进程
    fundmat = pd.DataFrame(index=corrmeanabs.index)
    #逐报告期计算因子股票池持仓
    for period in corrmeanabs.index:
        corrrow = corrmeanabs.ix[period,:].dropna().sort_values(ascending=False)
        #大类因子筛选
        if strategy.ix[mfid,'if_classify']:
            for kind in kinds:
                if len(pd.Series(factorkinds[kind])) > kindnum:
                    corrrow[factorkinds[kind]].dropna().sort_values(ascending=False).iloc[kindnum:len(corrrow[factorkinds[kind]].dropna())] = 0
        #计算位次因子股票池持仓
        factor = corrrow.index[i]
        direct = (corrmean.ix[period,factor]>0) * 1 + (corrmean.ix[period,factor]<0) * strategy.ix[mfid,'layer_num']
        try:
            stocks = list(position[period,factor,direct]) #因为数据旧导致2017年7.31有currentradio的std数据而8.31没有导致的8.31报except
        except Exception as e:
            print e
            stocks = ['SK.000001'] #随便写个吧
        for stock in stocks:
            fundmat.ix[period,stock] = 1.0/len(stocks)
        print '%s factorfund calculated: %s fund%s' %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'),str(i+1))
    #因子股票池持仓csv输出
    fundmat.to_csv('mf_sk_%s_fund%s.csv' %(mfid,str(i+1)))
    print mfid

def fundcsvall(): #导出多因子股票池持仓，拆为函数便于多进程
    fundall = pd.DataFrame(index=corrmeanabs.index)
    #逐报告期计算因子股票池持仓
    for period in corrmeanabs.index:
        corrrow = corrmeanabs.ix[period,:].dropna().sort_values(ascending=False)
        #大类因子筛选
        if strategy.ix[mfid,'if_classify']:
            for kind in kinds:
                if len(pd.Series(factorkinds[kind])) > kindnum:
                    corrrow[factorkinds[kind]].dropna().sort_values(ascending=False).iloc[kindnum:len(corrrow[factorkinds[kind]].dropna())] = 0
        #计算多因子股票池持仓
        if strategy.ix[mfid,'use_factor_method']: #为1即按百分比取，为0即按数值名次取
            factors = corrrow[corrrow>=np.percentile(corrrow,strategy.ix[mfid,'use_factor_num'])].index
        else:
            factors = corrrow[corrrow>=corrrow.iloc[strategy.ix[mfid,'use_factor_num']-1]].index
        fundallmat = pd.DataFrame(columns=factors)
        for factor in factors:
            direct = (corrmean.ix[period,factor]>0) * 1 + (corrmean.ix[period,factor]<0) * strategy.ix[mfid,'layer_num']
            try:
                stocks = list(position[period,factor,direct]) #因为数据旧导致2017年7.31有currentradio的std数据而8.31没有导致的8.31报except
            except Exception as e:
                print e
                stocks = ['SK.000001'] #随便写个吧
            for stock in stocks:
                fundallmat.ix[stock,factor] = 1.0/len(stocks)/len(factors)
        fundallsum = fundallmat.sum(axis=1)
        for stock in fundallsum.index:
            fundall.ix[period,stock] = fundallsum[stock]
        print '%s factorfund calculated: %s fundall' %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'))
    #因子股票池持仓csv输出
    fundall.to_csv('mf_sk_%s_fundall.csv' %mfid)

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

#读取多因子策略
command = "select * from mf_sk_paras"
strategy = pd.read_sql(command,conn_aa).set_index('mf_id')

#读取因子分类
command = "select factor_name,factor_kind from mf_sk_factors"
factorkinds = pd.read_sql(command,conn_aa).set_index('factor_kind').ix[:,'factor_name']
kinds = pd.Series(factorkinds.index).drop_duplicates()
'''
#读取股票因子持仓
command = "select periods_date,sk_code,factor_name,factor_position from mf_sk_factorvalue_std"
position = pd.read_sql(command,conn_aa).set_index(['periods_date','factor_name','factor_position']).ix[:,'sk_code']
print 'Data loaded.'
'''
#（不怎么需要改的）环境变量
fundnum = 4 #需要的位次因子股票池个数
kindnum = 2 #每类因子中限定的取用个数（如做大类因子限定）
csvpath = '/home/huyang/MF1710/' #当前持仓csv存储的绝对路径（即当前目录）
aapath = '/home/huyang/recommend_model/asset_allocation_v1/' #运行上载框架的目录
'''
#逐策略逐报告期算出因子股票池持仓
periodall = pd.Series(pd.date_range(start=datetime.datetime(1990,1,1),
                                    end=datetime.datetime.now(),freq='M'))
today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
for mfid in strategy.index:
    #生成报告期
    realstart = periodall[periodall.apply(lambda x: x.date())<=strategy.ix[mfid,'start_time']].iloc[-strategy.ix[mfid,'lookback_num']]
    periods = pd.date_range(start=realstart,end=datetime.datetime.now(),freq='M')
    #读取因子分层秩相关系数
    command = "select periods_date,factor_name,layer_corrs from mf_sk_layers where mf_id = '%s'" %mfid
    corrs = pd.read_sql(command,conn_aa).set_index(['periods_date','factor_name']).ix[:,'layer_corrs'].unstack('factor_name')
    corrmean = corrs.rolling(strategy.ix[mfid,'lookback_num']).mean().dropna(how='all')
    corrmeanabs = corrmean.abs()
    #启动多线程计算因子股票池持仓
    p = pd.Series(index=range(0,fundnum))
    for i in p.index:
        p[i] = multiprocessing.Process(target = fundcsvi, args = (i,))
        p[i].start()
    q = multiprocessing.Process(target = fundcsvall)
    q.start()
    for i in p.index:
        p[i].join()
    q.join()
    print 'All %s factorfund calculated.' %mfid
print 'All factorfund calculated.'
'''
#断开数据库
cur_aa.close()
conn_aa.commit()
conn_aa.close()

#生成portfolio文件
pcolumns = ['ra_portfolio_id','ra_portfolio_name','ra_portfolio_algo','mz_markowitz_id',
            'mz_markowitz_name','risk','mz_highlow_id','mz_highlow_name','mz_highlow_algo',
            'asset_id','pool_id','sum1','sum2','lower','upper','allocate_algo',
            'allocate_turnover_filter','highlow_turnover_filter','portfolio_turnover_filter',
            'allocate_date_type','allocate_adjust_position_period',
            'allocate_adjust_position_dates','allocate_look_back','allocate_wavelet',
            'start_date','end_date','timing_id','riskmgr_id','csv']
for mfid in strategy.index:
    #生成多因子股票池portfolio文件，占用MZ的首位1(hy使用号段)&次位0(因子策略)号码段
    #3,4位为两位策略号，对应MF号码4,5位；5位固定为0(表示多因子)；6位固定为0(以备高低风险使用)
    #如MF.000010对应MZ.100100
    mzid = '10%s00' %mfid[6:8]
    mzname = 'hy_%s_fundall' %mfid
    csvname = 'mf_sk_%s_fundall.csv' %mfid
    stocks = pd.read_csv(csvname,index_col=['periods_date']).columns
    portfolio = []
    for stock in stocks:
        prow = pd.Series(['PO.%s'%mzid,mzname,1,'MZ.%s'%mzid,
                          mzname,1,'HL.%s'%mzid,mzname,2,
                          stock,'',0,0,0,0,5,0,0,0,'trade_week',1,'',26,0,
                          datetime.datetime.strftime(strategy.ix[mfid,'start_time'],'%Y/%m/%d'),
                          '','','','%s%s'%(csvpath,csvname)],
                         index=pcolumns) #然而大部分参数没有什么用，重点在allocate_algo选5
        portfolio += [prow]
    portfolio = pd.concat(portfolio,axis=1).T
    portfolio.to_csv('%s.portfolio' %mzname,index=False)
    print '%s built.' %mzname
    #生成位次因子股票池portfolio文件，占用MZ的首位1(hy使用号段)&次位0(因子策略)号码段
    #3,4位为两位策略号，对应MF号码4,5位；5位为因子位次；6位固定为0(以备高低风险使用)
    #如MF.000010对应MZ.100110~MZ.100190(不一定用完)
    for i in range(0,fundnum):
        mzid = '10%s%s0' %(mfid[6:8],str(i+1))
        mzname = 'hy_%s_fund%s' %(mfid,str(i+1))
        csvname = 'mf_sk_%s_fund%s.csv' %(mfid,str(i+1))
        stocks = pd.read_csv(csvname,index_col=['periods_date']).columns
        portfolio = []
        for stock in stocks:
            prow = pd.Series(['PO.%s'%mzid,mzname,1,'MZ.%s'%mzid,
                              mzname,1,'HL.%s'%mzid,mzname,2,
                              stock,'',0,0,0,0,5,0,0,0,'trade_week',1,'',26,0,
                              datetime.datetime.strftime(strategy.ix[mfid,'start_time'],'%Y/%m/%d'),
                              '','','','%s%s'%(csvpath,csvname)],
                             index=pcolumns) #然而大部分参数没有什么用，重点在allocate_algo选5
            portfolio += [prow]
        portfolio = pd.concat(portfolio,axis=1).T
        portfolio.to_csv('%s.portfolio' %mzname,index=False)
        print '%s built.' %mzname
    print 'All %s protfolio bulit.' %mfid
print 'All protfolio bulit.'

#使用资产配置组合已有框架上载csv至数据库
os.system('') #调用可跑框架的环境
for mfid in strategy.index:
    mzid = '10%s00' %mfid[6:8]
    mzname = 'hy_%s_fundall' %mfid
    csvname = 'mf_sk_%s_fundall.csv' %mfid
    print os.system('. /home/jiaoyang/tsf/bin/activate && cd %s' %aapath \
                    + ' && python shell/roboadvisor.py util imp_portfolio --path %s' \
                    %('%s%s'%(csvpath,'%s.portfolio'%mzname)) \
                    + ' && python shell/roboadvisor.py markowitz pos --id MZ.%s --csv %s' \
                    %(mzid,'%s%s'%(csvpath,csvname)) \
                    + ' && python shell/roboadvisor.py markowitz nav --id MZ.%s' %mzid \
                    + ' && python shell/roboadvisor.py markowitz turnover --id MZ.%s' %mzid)
    for i in range(0,fundnum):
        mzid = '10%s%s0' %(mfid[6:8],str(i+1))
        mzname = 'hy_%s_fund%s' %(mfid,str(i+1))
        csvname = 'mf_sk_%s_fund%s.csv' %(mfid,str(i+1))
        print os.system('. /home/jiaoyang/tsf/bin/activate && cd %s' %aapath \
                        + ' && python shell/roboadvisor.py util imp_portfolio --path %s' \
                        %('%s%s'%(csvpath,'%s.portfolio'%mzname)) \
                        + ' && python shell/roboadvisor.py markowitz pos --id MZ.%s --csv %s' \
                        %(mzid,'%s%s'%(csvpath,csvname)) \
                        + ' && python shell/roboadvisor.py markowitz nav --id MZ.%s' %mzid \
                        + ' && python shell/roboadvisor.py markowitz turnover --id MZ.%s' %mzid)
    print 'All %s uploaded.' %mfid
print 'All uploaded.'
