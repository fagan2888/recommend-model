#coding=utf8
import pandas as pd
import numpy as np
import datetime,time
import MySQLdb
import multiprocessing
import os
import warnings
warnings.filterwarnings("ignore")

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

#读取基金多因子策略
command = "select * from mf_skfd_paras"
strategy_fd = pd.read_sql(command,conn_aa).set_index('mf_id')

#读取股票多因子策略
command = "select * from mf_sk_paras"
strategy_sk = pd.read_sql(command,conn_aa).set_index('mf_id')

#读取因子分类
command = "select factor_name,factor_kind from mf_sk_factors"
factorkinds = pd.read_sql(command,conn_aa).set_index('factor_kind').ix[:,'factor_name']
kinds = pd.Series(factorkinds.index).drop_duplicates()

#（不怎么需要改的）环境变量
kindnum = 2 #每类因子中限定的取用个数（如做大类因子限定）
csvpath = '/home/huyang/MF1710/' #当前持仓csv存储的绝对路径（即当前目录）
aapath = '/home/huyang/recommend_model/asset_allocation_v1/' #运行上载框架的目录

#逐策略逐报告期算出多因子基金池持仓
today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
for mfid in strategy_fd.index:
    skid = strategy_fd.ix[mfid,'mf_skid']
    #读取因子基金池
    command = "select date,factor_name,factor_end,fd_code from mf_skfd_factorfundpool where mf_id = '%s'" %mfid
    fundpool = pd.read_sql(command,conn_aa).set_index(['factor_name','factor_end'])
    periodrange = fundpool.ix[:,'date'].drop_duplicates()
    #生成报告期
    periods = pd.date_range(start=min(periodrange),end=max(periodrange),freq='M')
    #读取因子分层秩相关系数
    command = "select periods_date,factor_name,layer_corrs from mf_sk_layers where mf_id = '%s'" %skid
    corrs = pd.read_sql(command,conn_aa).set_index(['periods_date','factor_name']).ix[:,'layer_corrs'].unstack('factor_name')
    corrmean = corrs.rolling(strategy_sk.ix[skid,'lookback_num']).mean().dropna(how='all')
    corrmeanabs = corrmean.abs()
    #计算多因子基金池持仓
    fundall = pd.DataFrame(index=periods)
    #逐报告期计算多因子基金池持仓
    for period in periods:
        corrrow = corrmeanabs.ix[period,:].dropna().sort_values(ascending=False)
        #大类因子筛选
        if strategy_sk.ix[skid,'if_classify']:
            for kind in kinds:
                if len(pd.Series(factorkinds[kind])) > kindnum:
                    corrrow[factorkinds[kind]].dropna().sort_values(ascending=False).iloc[kindnum:len(corrrow[factorkinds[kind]].dropna())] = 0
        #计算多因子基金池持仓
        if strategy_sk.ix[skid,'use_factor_method']: #为1即按百分比取，为0即按数值名次取
            factors = corrrow[corrrow>=np.percentile(corrrow,strategy_sk.ix[skid,'use_factor_num'])].index
        else:
            factors = corrrow[corrrow>=corrrow.iloc[strategy_sk.ix[skid,'use_factor_num']-1]].index
        fundallmat = pd.DataFrame(columns=factors)
        for factor in factors:
            direct = (corrmean.ix[period,factor]>0) * 1
            factorperiods = fundpool.ix[factor].ix[direct].ix[:,'date'].drop_duplicates()
            factorperiod = factorperiods[factorperiods<=period.date()].max()
            try:
                funds = pd.Series(fundpool.ix[factor].ix[direct].set_index('date').ix[:,'fd_code'].ix[factorperiod].values) #因为前面没算出std及tv数据导致except或者funds为空
                for fund in funds:
                    fundallmat.ix[fund,factor] = 1.0/len(funds)
            except Exception as e:
                print period,e,factor
        fundallsum = fundallmat.sum(axis=1)
        fundallsum = fundallsum/fundallsum.sum()  #因为前面没算出std及tv数据导致需要重新规整下比例
        for fund in fundallsum.index:
            fundall.ix[period,fund] = fundallsum[fund]
        print '%s fundpool calculated: %s' %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'))
    #多因子基金池持仓csv输出
    fundall.to_csv('mf_skfd_%s.csv' %mfid)
    print 'All %s fundpool calculated.' %mfid
print 'All fundpool calculated.'

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
for mfid in strategy_fd.index:
    #生成多因子基金池portfolio文件，占用MZ的首位1(hy使用号段)&次位0(因子策略)号码段
    #3,4位为两位策略号，对应MF号码4,5位；5位固定为9(表示因子基金池)；6位固定为0(以备高低风险使用)
    #如MF.100010对应MZ.100190
    mzid = '10%s90' %mfid[6:8]
    mzname = 'hy_%s' %mfid
    csvname = 'mf_skfd_%s.csv' %mfid
    funds = pd.read_csv(csvname,index_col=0).columns
    portfolio = []
    for fund in funds:
        prow = pd.Series(['PO.%s'%mzid,mzname,1,'MZ.%s'%mzid,
                          mzname,1,'HL.%s'%mzid,mzname,2,
                          fund,'',0,0,0,0,5,0,0,0,'trade_week',1,'',26,0,
                          datetime.datetime.strftime(strategy_fd.ix[mfid,'start_time'],'%Y/%m/%d'),
                          '','','','%s%s'%(csvpath,csvname)],
                         index=pcolumns) #然而大部分参数没有什么用，重点在allocate_algo选5
        portfolio += [prow]
    portfolio = pd.concat(portfolio,axis=1).T
    portfolio.to_csv('%s.portfolio' %mzname,index=False)
    print '%s built.' %mzname
print 'All protfolio bulit.'

#使用资产配置组合已有框架上载csv至数据库
os.system('') #调用可跑框架的环境
for mfid in strategy_fd.index:
    mzid = '10%s90' %mfid[6:8]
    mzname = 'hy_%s' %mfid
    csvname = 'mf_skfd_%s.csv' %mfid
    print os.system('. /home/jiaoyang/tsf/bin/activate && cd %s' %aapath \
                    + ' && python shell/roboadvisor.py util imp_portfolio --path %s' \
                    %('%s%s'%(csvpath,'%s.portfolio'%mzname)) \
                    + ' && python shell/roboadvisor.py markowitz pos --id MZ.%s --csv %s' \
                    %(mzid,'%s%s'%(csvpath,csvname)) \
                    + ' && python shell/roboadvisor.py markowitz nav --id MZ.%s' %mzid \
                    + ' && python shell/roboadvisor.py markowitz turnover --id MZ.%s' %mzid)
    print '%s uploaded.' %mfid
print 'All uploaded.'
