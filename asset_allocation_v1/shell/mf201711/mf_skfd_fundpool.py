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

mf_base = {"host": "127.0.0.1",
             "port": 3306,
             "user": "huyang",
             "passwd": "uMXfKx03Blg3jdPmauMx",
             "db":"mofang",
             "charset": "utf8"}

conn_mf  = MySQLdb.connect(**mf_base)
cur_mf   = conn_mf.cursor(MySQLdb.cursors.DictCursor)
conn_mf.autocommit(True)

#读取基金多因子策略
command = "select * from mf_skfd_paras"
strategy = pd.read_sql(command,conn_aa).set_index('mf_id')
print 'Fund MF-strategy loaded.'

#读取基金持仓
command = "select fd_code,sk_code,report_date,position from mf_skfd_position"
positions = pd.read_sql(command,conn_aa).set_index(['report_date','fd_code','sk_code']).ix[:,'position']
print 'Fund position loaded.'

#读取因子列表
command = "select factor_name,factor_source from mf_sk_factors"
sources = pd.read_sql(command,conn_aa).set_index('factor_name').ix[:,'factor_source']
factors_market = list(sources[sources==0].index)
factors_report = list(sources[sources==1].index)
print "Factors' setting loaded."

#读取基金规模、机构持仓比例
command = "select report_date,size,hold,fd_code from mf_skfd_marketreport"
market = pd.read_sql(command,conn_aa)
hold = market.set_index(['report_date','fd_code']).ix[:,'hold'].unstack('fd_code')
size = market.set_index(['report_date','fd_code']).ix[:,'size'].unstack('fd_code')
print "Fund market report loaded."

#读取基金业绩
command = "select * from mf_skfd_derivereport"
derive = pd.read_sql(command,conn_aa).drop(['id','created_at','updated_at'],axis=1).set_index(['date','fd_code'])
print "Fund derive report loaded."

#读取基金交易日
alldays = pd.Series(pd.date_range(start=datetime.datetime(1990,1,1),
                                  end=datetime.datetime.now(),freq='D'))
today = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d')
command = "select ra_code,ra_date,ra_inc from ra_fund_nav where ra_type = 1" #ra_type=1筛选股票基金
tradedata = pd.read_sql(command,conn_mf).set_index(['ra_code','ra_date']).ix[:,'ra_inc'].unstack('ra_code')
tradedata = tradedata.ix[alldays,:] #使用自然日顺序规整数据
tradedata[tradedata==0] = None #涨跌为0即认为当日未有交易
tradedays = tradedata.rolling(500).count() #计算前500自然日内基金交易的天数
print "Fund tradedata loaded."
trade_valids = tradedays[tradedays>250] #固定挑选交易超过250日的基金为合法

#逐策略计算基金池
for mfid in strategy.index:
    #生成报告期
    periodall = pd.Series(pd.date_range(start=strategy.ix[mfid,'start_time'],
                                        end=datetime.datetime.now(),freq='M'))
    periodfund = pd.Series(pd.date_range(start=strategy.ix[mfid,'start_time'],
                                        end=datetime.datetime.now(),freq='6M')) #要求start_time必须是0630或1231
    #读取股票因子值
    command = "select periods_date,sk_code,factor_name,factor_value_std from mf_sk_factorvalue_std where mf_id = '%s'" %strategy.ix[mfid,'mf_skid']
    skvalue = pd.read_sql(command,conn_aa).set_index(['periods_date','factor_name','sk_code']).ix[:,'factor_value_std']
    print mfid,'stock factor value loaded.'
    #取用需要的业绩列
    achieve = derive.ix[:,strategy.ix[mfid,'alphakind']].unstack('fd_code')
    achieve = achieve.ix[alldays,:].fillna(method='pad').ix[periodall,:]
    #逐时期计算基金池
    remainpool = pd.Series([[]*len(sources)*2],index=map(lambda x: x+'_L', sources.index)+map(lambda x: x+'_S', sources.index)) #用于记录前期池
    for period in periodall:
        datestr = datetime.datetime.strftime(period,'%Y%m%d')
        if_period = True #是否在该时期做计算
        #计算基金因子暴露，只有每年3,4,8月做计算才对
        if datestr[4:8] == '0331': #用1231就知道的1231行情数据对应1231的基金持仓
            mk_fac = factors_market
            rp_fac = []
            mkdate = datetime.datetime.strptime(str(int(datestr)-9100),'%Y%m%d') #1231
            fddate = datetime.datetime.strptime(str(int(datestr)-9100),'%Y%m%d') #1231
        elif datestr[4:8] == '0430': #用0430才知道的1231财报数据对应1231的基金持仓
            mk_fac = []
            rp_fac = factors_report
            rpdate = datetime.datetime.strptime(datestr,'%Y%m%d') #0430
            fddate = datetime.datetime.strptime(str(int(datestr)-9199),'%Y%m%d') #1231
        elif datestr[4:8] == '0831': #用0630知道的0630行情数据和0831知道的0630财报数据对应0630的基金持仓
            mk_fac = factors_market
            rp_fac = factors_report
            mkdate = datetime.datetime.strptime(str(int(datestr)-201),'%Y%m%d') #0630
            rpdate = datetime.datetime.strptime(datestr,'%Y%m%d') #0831
            fddate = datetime.datetime.strptime(str(int(datestr)-201),'%Y%m%d') #0630
        else:
            mk_fac = []
            rp_fac = []
            if_period = False
        #计算基金合法性
        if if_period:
            trade_valid = tradedays.ix[period,:].dropna().index #基金交易日限制
            #此处有两种思路，因有交易数据的基金数与有机构持仓、规模、业绩数据的基金数相差甚远（多了N倍），
            #故思路一为取机构持仓、规模、业绩均合法的基金，思路二为在交易日合法的基金中去除机构持仓、规模、业绩不合法的基金，
            #因而思路二中没有机构持仓、规模、业绩数据的基金会默认直接过审。
            #原先使用的合法性验证规则是思路二，这里暂且按思路一试试
            #思路一会有valid为空的时间（16年6月期）的！还是用思路二吧
            #思路二在16年6月期也有got str的问题！看来是数据有问题。
            #检查到了有std、tv因子数据没有算出，改计算公式为rollingstd,rollingsum后已经修复了。
            '''
            #思路一
            hold_valid = hold.ix[fddate,trade_valid].dropna() #计算的基础是交易日中合法的股票
            hold_valid = (hold_valid - hold_valid.mean()) / hold_valid.std() #计算为标准化数据
            hold_valid = hold_valid[hold_valid>=strategy.ix[mfid,'holdlimit']].index #机构持仓限制，保留持仓高的
            size_valid = size.ix[fddate,trade_valid].dropna() #计算的基础是交易日中合法的股票
            size_valid = (size_valid - size_valid.mean()) / size_valid.std() #计算为标准化数据
            size_valid = size_valid[size_valid<=strategy.ix[mfid,'sizelimit']].index #基金规模限制，保留规模小的
            alpha_valid = achieve.ix[period,trade_valid].dropna() #计算的基础是交易日中合法的股票
            alpha_valid = (alpha_valid - alpha_valid.mean()) / alpha_valid.std() #计算为标准化数据
            alpha_valid = alpha_valid[alpha_valid>=strategy.ix[mfid,'alphalimit']].index #基金业绩限制，保留业绩高的
            valid = list(set(hold_valid)&set(size_valid)&set(alpha_valid)) #取并集限定合法性
            valid = map(lambda x: 'FD.'+str(x), valid) #编号方式暂时还不同
            '''
            #思路二
            hold_invalid = hold.ix[fddate,trade_valid].dropna() #计算的基础是交易日中合法的股票
            hold_invalid = (hold_invalid - hold_invalid.mean()) / hold_invalid.std() #计算为标准化数据
            hold_invalid = hold_invalid[hold_invalid < strategy.ix[mfid,'holdlimit']].index #机构持仓限制，去掉持仓低的
            size_invalid = size.ix[fddate,trade_valid].dropna() #计算的基础是交易日中合法的股票
            size_invalid = (size_invalid - size_invalid.mean()) / size_invalid.std() #计算为标准化数据
            size_invalid = size_invalid[size_invalid > strategy.ix[mfid,'sizelimit']].index #基金规模限制，去掉规模大的
            alpha_invalid = achieve.ix[period,trade_valid].dropna() #计算的基础是交易日中合法的股票
            alpha_invalid = (alpha_invalid - alpha_invalid.mean()) / alpha_invalid.std() #计算为标准化数据
            alpha_invalid = alpha_invalid[alpha_invalid < strategy.ix[mfid,'alphalimit']].index #基金业绩限制，去掉业绩低的
            valid = list(set(trade_valid)-set(hold_invalid)-set(size_invalid)-set(alpha_invalid)) #取差集限定合法性
            valid = map(lambda x: 'FD.'+str(x), valid) #编号方式暂时还不同
            #'''
        #逐因子计算基金池
        for factor in mk_fac: #行情因子（仅有引用的数据时间不同）
            try: #因为种种原因有些因子数据没有算出，故加个try
                fund_factorvalue = (positions.ix[fddate].ix[valid] * skvalue.ix[mkdate].ix[factor]).unstack('sk_code').sum(axis=1).dropna()
                top_L = fund_factorvalue.sort_values(ascending=False).iloc[0:strategy.ix[mfid,'sparepoolsize']] #算出二级池
                top_S = fund_factorvalue.sort_values(ascending=True).iloc[0:strategy.ix[mfid,'sparepoolsize']]
                remain_L = top_L.ix[remainpool[factor+'_L']].dropna().index #计算仍保留在二级池的基金
                remain_S = top_S.ix[remainpool[factor+'_S']].dropna().index
                pool_L = pd.Series(list(remain_L)+list(top_L.index)).drop_duplicates().iloc[0:strategy.ix[mfid,'poolsize']] #计算当期基金池
                pool_S = pd.Series(list(remain_S)+list(top_S.index)).drop_duplicates().iloc[0:strategy.ix[mfid,'poolsize']]
                remainpool[factor+'_L'] = list(pool_L) #更新前期池记录
                remainpool[factor+'_S'] = list(pool_S) #更新前期池记录
                #写入数据库
                for fund in pool_L:
                    command = "insert into mf_skfd_factorfundpool (mf_id,date,factor_name,factor_end,fd_code,created_at,updated_at) " \
                              + " values ('%s','%s','%s',%s,'%s','%s','%s')" \
                              %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'),
                                factor,'1',fund,today,today)
                    cur_aa.execute(command)
                for fund in pool_S:
                    command = "insert into mf_skfd_factorfundpool (mf_id,date,factor_name,factor_end,fd_code,created_at,updated_at) " \
                              + " values ('%s','%s','%s',%s,'%s','%s','%s')" \
                              %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'),
                                factor,'0',fund,today,today)
                    cur_aa.execute(command)
                print mfid,period,'fundpool calculated: ',factor
            except Exception as e:
                print e
        for factor in rp_fac: #财报因子（仅有引用的数据时间不同）
            try: #因为种种原因有些因子数据没有算出，故加个try
                fund_factorvalue = (positions.ix[fddate].ix[valid] * skvalue.ix[rpdate].ix[factor]).unstack('sk_code').sum(axis=1).dropna()
                top_L = fund_factorvalue.sort_values(ascending=False).iloc[0:strategy.ix[mfid,'sparepoolsize']] #算出二级池
                top_S = fund_factorvalue.sort_values(ascending=True).iloc[0:strategy.ix[mfid,'sparepoolsize']]
                remain_L = top_L.ix[remainpool[factor+'_L']].dropna().index #计算仍保留在二级池的基金
                remain_S = top_S.ix[remainpool[factor+'_S']].dropna().index
                pool_L = pd.Series(list(remain_L)+list(top_L.index)).drop_duplicates().iloc[0:strategy.ix[mfid,'poolsize']] #计算当期基金池
                pool_S = pd.Series(list(remain_S)+list(top_S.index)).drop_duplicates().iloc[0:strategy.ix[mfid,'poolsize']]
                remainpool[factor+'_L'] = list(pool_L) #更新前期池记录
                remainpool[factor+'_S'] = list(pool_S) #更新前期池记录
                #写入数据库
                for fund in pool_L:
                    command = "insert into mf_skfd_factorfundpool (mf_id,date,factor_name,factor_end,fd_code,created_at,updated_at) " \
                              + " values ('%s','%s','%s',%s,'%s','%s','%s')" \
                              %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'),
                                factor,'1',fund,today,today)
                    cur_aa.execute(command)
                for fund in pool_S:
                    command = "insert into mf_skfd_factorfundpool (mf_id,date,factor_name,factor_end,fd_code,created_at,updated_at) " \
                              + " values ('%s','%s','%s',%s,'%s','%s','%s')" \
                              %(mfid,datetime.datetime.strftime(period,'%Y-%m-%d'),
                                factor,'0',fund,today,today)
                    cur_aa.execute(command)
                print mfid,period,'fundpool calculated:',factor
            except Exception as e:
                print e
    print mfid,'fundpool all calculated.'

#断开数据库
cur_mf.close()
conn_mf.commit()
conn_mf.close()

cur_aa.close()
conn_aa.commit()
conn_aa.close()
