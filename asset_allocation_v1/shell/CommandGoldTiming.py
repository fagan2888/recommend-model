# coding=utf-8
import sys
import click
sys.path.append('shell')
#from db import *
import logging
import pandas as pd
import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from ipdb import set_trace

import config
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
import trade_date
from calendar import monthrange
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)

from sqlalchemy import MetaData,Table,select
from sklearn import datasets,linear_model

##############################################################################

@click.group(invoke_without_command=True)
@click.pass_context
def gt(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(gold_view_update)
    else:
        pass

##############################################################################

def load_gold_indicator():
    feature_names = {
        'MC.GD0013':'LD_sg',
        'MC.GD0014':'USrdi',
        'MC.GD0015':'UScpi',
        'MC.GD0016':'USrty',
        'MC.GD0017':'USndi',
        'MC.GD0018':'USnrty',
        'MC.GD0019': 'comex_uncom_long',
        'MC.GD0020': 'comex_uncom_short',
        'MC.GD0021': 'comex_com_long',
        'MC.GD0022': 'comex_com_short',
        'MC.GD0023': 'comex_inventory',
        'MC.GD0024': 'comex_Pos',
        'MC.GD0025': 'SPDR_Pos',
        'MC.GD0026': 'iShares_vol',
        'MC.GD0027': 'SPDR_vol',
        'MC.GD0028': 'comex_price',
        'MC.GD0029': 'SPDR_price',
        'MC.GD0030': 'iShares_price',
    }
    '''
    'LD_sg' 伦敦现货黄金价格
    'USrdi' 实际美元指数
    ‘UScpi' 美国核心CPI
    ’USrty‘ 美国10年期国债实际收益率
    'USndi' 名义美元指数
    'USnrty'美国10年期国债收益率
    '''
    engine = database.connection('wind')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql = session.query(
        mc_gold_indicator.globalid,
        mc_gold_indicator.mc_gold_date,
        mc_gold_indicator.mc_gold_value,
        ).filter(mc_gold_indicator.globalid.in_(feature_names.keys())).statement

    gdi = pd.read_sql(
        sql,
        session.bind,
        index_col = ['mc_gold_date', 'globalid'],
        parse_dates = ['mc_gold_date'],
    )

    session.commit()
    session.close()

    gdi = gdi.unstack()
    gdi.columns = gdi.columns.levels[1]
    gdi = gdi.rename(columns = feature_names)

    return gdi

#更新数据库函数
def save(gid, df):
    fmt_columns = ['globalid','mc_gold_date','mc_gold_value','created_at' 'updated_at']
    fmt_precision = 6
    if not df.empty:
        df = database.number_format(df, fmt_columns, fmt_precision)
    # 保存到数据库
    db = database.connection('lcmf_wind')
    t2 = Table('mc_gold_indicator', MetaData(bind=db), autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns, (t2.c.globalid == gid))
    df_old = pd.read_sql(s, db, index_col=['globalid', 'mc_gold_date'], parse_dates=['mc_gold_date'])
    if not df_old.empty:
        df_old = database.number_format(df_old, fmt_columns, fmt_precision)
    # 更新数据库
    database.batch(db, t2, df, df_old, timestamp=False)

#更新数据库
def mc_gold_indicator_save():
    pass


##############################################################################

# 当月日期转换为下月
def next_month(now_time):
    '''
    # 输入：字符串如 ‘2018-06’
    # 输出：‘2018-7’
    '''
    next_year = int(now_time[0:4])
    next_month = int(now_time[5:7]) + 1
    if next_month == 13:
        next_month = 1
        next_year += 1
    month_time = str(next_year) + '-' + str(next_month)
    return month_time

# 指数移动平均
def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s
    """
    #s = np.array(s)
    ema = []
    j = 1
    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)
    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(((s[n] - sma)*multiplier) + sma)
    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ((i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    ema1 = []
    for m in range(1,n):
        n = s[:m]
        ema1.append(np.mean(n))
    ema = ema1 + ema
    return np.array(ema)

# 数据平滑
def smooth(df,para=0.34,win_obs=7,para_obs=0):
    #df第1列必须为观点数据,df['data','','',...]
    
    df_data = df.iloc[:,0]
    data = df.loc[:,['USrdi','USrty']]#USrdi和USrty数据
    #
    data_ema = ema(df_data.values,win_obs)
    df['gold_smooth'] = ema(df_data.values,win_obs)
    # 构建观点强度值
    strength = df['gold_smooth'].pct_change().fillna(0).values/para*100

    ##### 增加利率和汇率同向的经验过滤###############################################
    if para_obs==0:
        view_amend = np.zeros(len(strength))
    else:
        #data[['USrdi','USrty']] = data[['USrdi','USrty']].shift(1).fillna(method='bfill')
        rdi_ema = pd.DataFrame(ema(data[['USrdi']].values,para_obs),index=data.index)
        rty_ema = pd.DataFrame(ema(data[['USrty']].values,para_obs),index=data.index)
        rdi_pct = rdi_ema.pct_change().fillna(0)
        rty_pct = rty_ema.pct_change().fillna(0)
        rdi_rty = rdi_pct*rty_pct
        factor = np.where(rdi_rty > 0,rdi_rty*10000.0,0)#去除负值
        symbol = np.where(rdi_pct >0 ,1.0,-1.0)#保留符号
        factor = factor*symbol*-1.0#因素1
        factor[np.abs(factor)<0.01] = 0
        factor1 = pd.DataFrame(factor).rolling(win_obs).mean().fillna(method='bfill').values

        factor_symbol1 = np.where(factor1==0,0.0,np.where(factor1> 0,1.0,-1.0)).flatten()#滚动值处理成符号[111110000-1-1-1-1]
        factor_symbol2 = np.where(factor == 0,0.0,1.0).flatten()#处理成【00001111】
        #对factor_symbol2做处理：往前数5个如果没有同号的，当前值设为1
        factor_symbol3 = factor_symbol2.copy()
        mn = factor.flatten()
        mn[mn < 0] = -1.0
        mn[mn > 0] = 1.0
        for i in range(5,len(factor_symbol2)):
            if mn[i] == 1:
                if list(mn[i-5:i]).count(1.0) < 1:
                    factor_symbol3[i] = 0.0
            elif mn[i] == -1:
                if list(mn[i-5:i]).count(-1.0) < 1:
                    factor_symbol3[i] = 0.0
        #print '############factor+factor_symbol2############/n',pd.DataFrame(np.c_[factor.flatten(),factor_symbol2,factor_symbol3])
        view_amend = factor_symbol1*factor_symbol3#对观点的符号修改

    data_strength = []
    for j in range(len(strength)):
        if view_amend[j] == 0.0:
            data_strength.append(strength[j])
        elif view_amend[j] > 0.0:
            data_strength.append(np.abs(strength[j]))
        elif view_amend[j] < 0.0:
            data_strength.append(np.abs(strength[j])*-1.0)

    #################################################################################

     ## 经验参数过滤
    para_strength = para*5
    for i in range(3,len(data_strength)):
        if (data_strength[i-1]*data_strength[i-2]>=0 and data_strength[i-1]*data_strength[i]<=0 and \
            abs(data_strength[i]) < para_strength):
            data_strength[i] = -1.0*data_strength[i]

    #print '###########data_strength##########',pd.DataFrame(data_strength)

    return data_strength,df['gold_smooth']

# 观点评价
def evaluation(df,para=0):
    #####df有两列，df['标的资产','观点']
    df_asset = df.iloc[:,0]
    df_view = df.iloc[:,1]
    #净值
    income = df_asset.pct_change().fillna(0).values * df_view.values
    strength_profit = (income/100+1).cumprod()#假设每月投资观点强度值%的比例，得到每月的收益，累积可计算出净值，期初净值为1.0000
    #方向准确率
    precision = np.where(income > para,np.ones(len(income)),np.zeros(len(income)))
    #precision = np.where(precision <= para,income,np.zeros(len(income)))
    precision_ratio = [0.5]
    for j in range(1,len(precision)):
        precision_ratio.append(precision[0:j].sum()/j)

    #4个月周期稳定率
    view = df_view.values
    win4M = []
    win4M_ratio = []
    for i in range(len(precision)):
        if i<4:
            win4M.append(1.0)
            win4M_ratio.append(1.0)
        else:
            if view[i]>=0.0 and view[i-1]>=0.0 and view[i-3]>=0.0 and view[i-4]>=0.0:
                win4M.append(1.0)
            elif i>=4 and view[i]<=0.0 and view[i-1]<=0.0 and view[i-3]<=0.0 and view[i-4]<=0.0:
                win4M.append(1.0)
            else:
                win4M.append(0.0)
            win4M_ratio.append(1.0*sum(win4M)/(i+1))

    return precision_ratio,strength_profit,win4M_ratio

# 测试参数
def para_test():
    data_para = []
    for i in range(1):
        for j in range(6,8):
            for m in range(6,8):
                data = cal_gold_view(para_IR=i,obs1=j,obs2=m)
                data = data[['precision','net','M4_ratio','precision_b','net_b','M4_ratio_b']]#测试参数用
                data_para.append([i,j,m] + list(data.iloc[-1,:].values))#将array转成list
    df_data = pd.DataFrame(np.array(data_para),columns=[['i','j','m','pre','net','M4','preb','netb','M4b']])#
    print df_data
    df_data.to_csv('gold_paratest.csv')
    return df_data

# 输出结果
def result():
    for i in range(1):
        for j in range(6,8):
            for m in range(6,8):
                data = cal_gold_view(para_IR=i,obs1=j,obs2=m)
                filename = '%s%s%s%s%s%s%s' % ('IR',str(i),'_win',str(j),'m',str(m),'.csv')
                data.to_csv(filename)

    return data

def cal_gold_view(para_IR=0,obs1=7,obs2=7):
    #@para
    #para_IR = 0,1,2
    #obs1 = 6,7,obs2=0,7

    #####
    # 1、导入数据，分别得到月频和日频数据集：M_data D_data
    data = load_gold_indicator()
    #print data
    M_data = data.loc[:,['UScpi']].dropna()
    M_data = M_data[~M_data.iloc[:,0].isin([0])]
    M_data = M_data.truncate(before = '1997-07-01')#获取该日期之后的数据

    ######
    # 2、获取real_dataset和forcast_dataset数据集@para_IR
    a_trade_date = trade_date.ATradeDate()
    index_month = a_trade_date.month_trade_date(begin_date = '1997-07-01')
    vecdict=dict(zip(M_data.index,index_month))#将index月末日期重命名为交易日期
    M_data = M_data.rename(index=vecdict)
    data_month = data.loc[:,['LD_sg','USnrty','USndi']].reindex(index_month)
    data_month = pd.concat([data_month,M_data],axis=1,join='inner')
    data_month['UScpi_ratio'] = data_month.loc[:,['UScpi']].pct_change(12).fillna(method='bfill')*100
    data_month = data_month.fillna(method='pad')
    ### 获取real_dataset
    real_dataset = data_month.loc[:,['UScpi_ratio']].copy()
    real_dataset['USrdi'] = data_month.loc[:,'USndi'] / data_month.loc[:,'UScpi']
    real_dataset['USrty'] = data_month.loc[:,'USnrty'] - data_month.loc[:,'UScpi_ratio']
    real_dataset['LD_sg'] = data_month.loc[:,'LD_sg'] / data_month.loc[:,'UScpi']
    real_dataset['UScpi'] = data_month.loc[:,'UScpi']
    real_dataset.drop(['UScpi_ratio'],axis=1,inplace=True)
    if para_IR == 0:#利率数据不处理
        pass
    elif para_IR == 1:#利率数据处理成零
        real_dataset['USrty'] =  np.where(real_dataset['USrty'] < 0 ,0,real_dataset['USrty'])#将真实利率小于零的位置赋值为0
    elif para_IR == 2:#利率数处理成相反数
        real_dataset['USrty'] =  np.where(real_dataset['USrty'] < 0 ,-real_dataset['USrty'],real_dataset['USrty'])#将真实利率小于零的位置赋值为相反数
    else:
        pass

    #####
    #3、回归系数<选择120个数滚动回归>，同时做因变量预测
    #real_dataset['USrdi','USrty','LD_sg','UScpi']
    linreg = linear_model.LinearRegression()
    win_rolling = 120
    forcast_gold = pd.DataFrame()
    for i in range(len(real_dataset)-win_rolling):
        x_par_month = real_dataset.iloc[0:i+win_rolling,0:2]
        y_par_month = real_dataset.iloc[0:i+win_rolling,2]
        model = linreg.fit(x_par_month,y_par_month)
        max_strength = max(y_par_month.pct_change().fillna(0).values) - min(y_par_month.pct_change().fillna(0).values)#便于强度值刻画
        #max_strength.append(max_strength)
        # 获取预测值
        m = real_dataset.index[i+win_rolling-1].strftime('%Y-%m')
        next_m = next_month(m)
        x_test = real_dataset[next_m]
        y_pred = linreg.predict(x_test[['USrdi','USrty']])
        index_x = x_test.index
        y_r = pd.DataFrame(y_pred,index=index_x,columns=['gold_real'])
        y = pd.DataFrame(y_pred*x_test['UScpi'].values,index=index_x,columns=['gold_forcast'])
        intercept_ = pd.DataFrame(linreg.intercept_,index=index_x,columns=['intercept'])
        coef_rdi = pd.DataFrame(np.ones(len(y_pred))*linreg.coef_[0],index=index_x,columns=['coef_USrdi'])
        coef_rty = pd.DataFrame(np.ones(len(y_pred))*linreg.coef_[1],index=index_x,columns=['coef_USrty'])
        zscore = pd.DataFrame(np.ones(len(y_pred))*model.score(x_par_month,y_par_month),index=index_x,columns=['zscore'])
        max_s = pd.DataFrame(np.ones(len(y_pred))*max_strength,index_x,columns=['max_s'])
        result = pd.concat([y_r,y,intercept_,coef_rdi,coef_rty,zscore,x_test[['USrdi','USrty','UScpi']],max_s],axis=1)
        forcast_gold = forcast_gold.append(result)
    forcast_gold = pd.concat([forcast_gold,data.loc[:,'LD_sg']],axis=1,join_axes=[forcast_gold.index])
    #print forcast_gold

    #####
    #4、结果
    #输出观点值及准确率/强度收益@para_win_obs
    gold_strength,gold_smooth = smooth(forcast_gold[['gold_forcast','USrdi','USrty']],para=0.34,win_obs=obs1,para_obs=obs2)
    forcast_gold['view_gold'] = gold_strength
    forcast_gold['smooth'] = gold_smooth
    gold_strength_b,gold_smooth = smooth(forcast_gold[['gold_real','USrdi','USrty']],para=0.36,win_obs=obs1,para_obs=obs2)
    forcast_gold['view_gold_b'] = gold_strength_b
    now = datetime.now()
    today_date = datetime(now.year,now.month,now.day)
    forcast_gold.loc[today_date] = None #增加一行
    forcast_gold[['view_gold','view_gold_b']] = forcast_gold[['view_gold','view_gold_b']].shift(1).fillna(0)#对观点值往后移动一位以匹配日期
    forcast_gold[['gold_forcast','gold_real','smooth','max_s','coef_USrdi','coef_USrty','intercept','zscore']] = \
        forcast_gold[['gold_forcast','gold_real','smooth','max_s','coef_USrdi','coef_USrty','intercept','zscore']].shift(1)
    #分析评估：方向准确率及强度择时收益及4个月周期稳定率
    precision_ratio,net_value,M4_ratio = evaluation(forcast_gold[['LD_sg','view_gold']],para=0)
    precision_ratio_b,net_value_b,M4_ratio_b = evaluation(forcast_gold[['LD_sg','view_gold_b']],para=0)
    forcast_gold['precision'] = precision_ratio
    forcast_gold['net'] = net_value
    forcast_gold['M4_ratio'] = M4_ratio
    forcast_gold['precision_b'] = precision_ratio
    forcast_gold['net_b'] = net_value_b
    forcast_gold['M4_ratio_b'] = M4_ratio_b
    #输出结果
    forcast_gold = forcast_gold[['gold_forcast','gold_real','LD_sg','smooth',\
        'view_gold','view_gold_b','precision','net','M4_ratio','precision_b','net_b','M4_ratio_b',\
        'UScpi','USrdi','USrty','coef_USrdi','coef_USrty','intercept','zscore']]
    #print forcast_gold
    return forcast_gold

##############################################################################

@gt.command()
@click.option('--start-date', 'startdate', default='2003-01-01', help=u'start date to calc')
@click.option('--end-date', 'enddate', default=datetime.today().strftime('%Y-%m-%d'), help=u'start date to calc')
@click.option('--viewid', 'viewid', default='MC.VW0006', help=u'macro timing view id')
@click.pass_context
def gold_view_update(ctx, startdate, enddate, viewid):
    #backtest_interval = pd.date_range(startdate, enddate)
    mv = cal_gold_view()
    today = datetime.now()
    union_mv = {}
    union_mv['mc_view_id'] = np.repeat(viewid,len(mv))
    union_mv['mc_date'] = mv.index
    union_mv['mc_inc'] = mv.view_gold
    union_mv['created_at'] = np.repeat(today,len(mv))
    union_mv['updated_at'] = np.repeat(today,len(mv))
    union_mv_df = pd.DataFrame(union_mv, columns = ['mc_view_id', 'mc_date', 'mc_inc', 'created_at', 'updated_at'])
    df_new = union_mv_df.set_index(['mc_view_id', 'mc_date'])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('mc_view_strength', metadata, autoload = True)
    columns = [
        t.c.mc_view_id,
        t.c.mc_date,
        t.c.mc_inc,
        t.c.created_at,
        t.c.updated_at,
    ]
    s = select(columns, (t.c.mc_view_id == viewid))
    df_old = pd.read_sql(s, db, index_col = ['mc_view_id', 'mc_date'], parse_dates = ['mc_date'])
    database.batch(db, t, df_new, df_old, timestamp = False)

##############################################################################
#########################

if __name__ == '__main__':
    view = cal_gold_view()
    print view
    view.to_csv('gold_view.csv')
    #print view['view_gold']
    #####
    #黄金持仓观点
    #pos_data = data.loc[:,['comex_uncom_long','comex_uncom_short','comex_com_long','comex_com_short','comex_Pos']].truncate(before = '1997-07-01')
    #print pos_data


