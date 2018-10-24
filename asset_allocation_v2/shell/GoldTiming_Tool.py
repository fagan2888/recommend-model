#coding = utf-8
import sys
sys.path.append('shell/')

import click
import pandas as pd
import numpy as np
import config
import trade_date
import logging
logger = logging.getLogger(__name__)
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData,Table,select
from db import database,asset_trade_dates,base_ra_index_nav
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime,timedelta
from sklearn import linear_model
from ipdb import set_trace

@click.group(invoke_without_command=True)
@click.pass_context
def gt(ctx):
    #function:...
    if ctx.invoked_subcommand is None:
        ctx.invoke(gold_view_update)
    else:
        pass

def load_gold_indicator():
    #function:读取数据
    feature_names = {
        'MC.GD0013':'LD_sg',
        'MC.GD0015':'UScpi',
        'MC.GD0017':'USndi',
        'MC.GD0018':'USnrty',
        'MC.GD0027':'comex_pos_fundlong',
        'MC.GD0028':'comex_pos_fundshort',
        }
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
        index_col = ['mc_gold_date','globalid'],
        parse_dates = ['mc_gold_date'],
    )
    session.commit()
    session.close()
    gdi = gdi.unstack()
    gdi.columns = gdi.columns.levels[1]
    gdi = gdi.rename(columns = feature_names)
    return gdi

def save(gid,df):
    #function:"更新数据"
    db = database.connection('wind')
    t = Table('mc_gold_indicator',MetaData(bind=db),autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns).where(t.c.globalid==gid)
    df_old = pd.read_sql(s,db,index_col=['globalid','mc_gold_date'],parse_dates=['mc_gold_date'])
    database.batch(db,t,df,df_old,timestamp=False)

def initial_update():
    #function:"更新黄金数据库"
    feature_names = {
        'MC.GD0013':'LD_sg',
        'MC.GD0015':'UScpi',
        'MC.GD0017':'USndi',
        'MC.GD0018':'USnrty',
        'MC.GD0027':'comex_pos_fundlong',
        'MC.GD0028':'comex_pos_fundshort',
    }

    today = datetime.now().strftime('%Y-%m-%d')
    df1 = pd.read_csv('goldprice_windupdate.csv').set_index(['date'])
    df2 = pd.read_csv('goldvariable_windupdate.csv').set_index(['date'])
    sorted_keys = sorted(feature_names.keys())
    #print(sorted_keys)
    for i in range(0,len(sorted_keys)):
        key_i = sorted_keys[i]
        columns_name = feature_names[key_i]
        if key_i == 'MC.GD0013':
            df = df1
        else:
            df = df2.loc[:,[columns_name]]

        df.index.name = 'mc_gold_date'
        df.columns = ['mc_gold_value']
        df.loc[:,'globalid'] = key_i
        df.loc[:,'created_at'] = today
        df.loc[:,'updated_at'] = today
        df = df.reset_index().set_index(['globalid','mc_gold_date'])
        save(key_i,df)

def next_month(now_time):
    #function:"获取下一月份"
    #input:eg,"2018-06"
    #output:eg,"2018-7"
    next_year = int(now_time[0:4])
    next_month = int(now_time[5:7]) + 1
    if next_month == 13:
        next_month = 1
        next_year += 1

    month_time = str(next_year) + '-' + str(next_month)

    return month_time

def ema(s,n):
    #input: array
    ema = []
    #获取第一期EMA值
    sma = sum(s[:n])/n
    para = 2 / float(1+n)
    ema.append(sma)
    ema.append(((s[n] - sma)*para) + sma)
    #计算EMA剩余期限的EMA值<循环>
    j = 1
    for i in s[n+1:]:
        tmp = ((i - ema[j]) * para) + ema[j]
        j = j+1
        ema.append(tmp)

    ema1 = [s[:x].mean() for x in range(1,n)]
    ema = ema1 + ema

    return np.array(ema)

def cal_cot(df,cot_win=30):
    #function:"将持仓数据处理成cot"
    df_cot = pd.DataFrame()
    df_cot['max'] = df.rolling(min_periods=1,window=cot_win).max()
    df_cot['min'] = df.rolling(min_periods=1,window=cot_win).min()
    df_cot['cot'] = (df - df_cot['min'])/(df_cot['max'] - df_cot['min'])

    return df_cot['cot'].fillna(0).values

def filter_fuc(values,para):
    #function:"简单过滤"
    filter_values = values
    for i in range(len(values)):
        if i > 2 and (values[i-1]*values[i-2] >= 0 and values[i-1]*values[i] <= 0 and abs(values[i]) < para):
            filter_values[i] = -1.0*values[i]
    return filter_values

def cal_view(df,para=0.34,win_obs=7,para_obs=7):
    #function:"计算观点"
    df_data = df.iloc[:,0]#观点数据
    data = df.loc[:,['USrdi','USrty']]
    df['gold_ema'] = ema(df_data.values,win_obs)
    strength = filter_fuc(df['gold_ema'].pct_change().fillna(0).values/para*100,para*5)#经验参>数过滤
    ####增加利率和汇率同向的经验过滤#################################
    if para_obs==0:
        view_amend = np.zeros(len(strength))
    else:
        rdi = pd.DataFrame(ema(data[['USrdi']].values,para_obs),index=data.index).pct_change().fillna(0)
        rty = pd.DataFrame(ema(data[['USrty']].values,para_obs),index=data.index).pct_change().fillna(0)
        rdi_rty = rdi*rty 
        factor = np.where(rdi_rty > 0,rdi_rty*10000.0,0)#去除负值 
        symbol = np.where(rdi > 0,1.0,-1.0)#保留符号
        factor = factor*symbol*-1.0#因素1
        factor[np.abs(factor)<0.01] = 0
        factor1 = pd.DataFrame(factor).rolling(win_obs).mean().fillna(method='bfill').values
        factor_symbol1 = np.where(factor1==0,0.0,np.where(factor1>0,1.0,-1.0)).flatten()#滚动值>处理成符号[111110000-1-1-1-1]
        factor_symbol2 = np.where(factor==0,0.0,1.0).flatten()#处理成[0000011111]
        #对factor_symbol2做处理：往前数4个，如果没有同号的，当前值设为1
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
        view_amend = factor_symbol1*factor_symbol3#对观点的符号修改
    data_strength = []
    for j in range(len(strength)):
        if view_amend[j] == 0.0:
            data_strength.append(strength[j])
        elif view_amend[j] > 0.0:
            data_strength.append(np.abs(strength[j]))
        elif view_amend[j] < 0.0:
            data_strength.append(np.abs(strength[j])*-1.0)

    return data_strength,df['gold_ema']

###############    主体函数     #####################

def cal_gold_view(obs1=7,cot_win=30,ema_win=12):
    #function:"黄金观点"
    #initial_update() #手动导入数据到数据库，不用csv数据读入
    data = load_gold_indicator()

    M_data = data.loc[:,['UScpi']].dropna()
    M_data = M_data[~M_data.iloc[:,0].isin([0])]
    M_data = M_data.truncate(before='1997-07-01')#获取该日期之后的日期
    a_trade_date = trade_date.ATradeDate()
    index_month = a_trade_date.month_trade_date(begin_date = '1997-07-01')
    vecdict = dict(zip(M_data.index,index_month))
    M_data = M_data.rename(index=vecdict)
    data_month = data.loc[:,['LD_sg','USnrty','USndi']].reindex(index_month)
    data_month = pd.concat([data_month,M_data],axis=1,join='inner')
    data_month[data_month.isin([0])] = np.nan
    data_month.fillna(method='pad',inplace=True)
    data_month['UScpi_ratio'] = data_month.loc[:,['UScpi']].pct_change(12).fillna(method='bfill')*100
    data_month = data_month.fillna(method='pad')
    data_comex = data.loc[:,['comex_pos_fundlong','comex_pos_fundshort']].replace(0,np.nan).fillna(method='bfill').reindex(index=data_month.index)
    filter_para = 0.026#经验参数
    netcot = ema(cal_cot(data_comex.iloc[:,0],cot_win)-cal_cot(data_comex.iloc[:,1],cot_win),ema_win)
    view_cot = filter_fuc(netcot,filter_para)
    real_data = data_month.loc[:,['UScpi_ratio']].copy()
    real_data['USrdi'] = data_month.loc[:,'USndi'] / data_month.loc[:,'UScpi']
    real_data['USrty'] = data_month.loc[:,'USnrty'] - data_month.loc[:,'UScpi_ratio']
    real_data['LD_sg'] = data_month.loc[:,'LD_sg'] / data_month.loc[:,'UScpi']
    real_data['UScpi'] = data_month.loc[:,'UScpi']
    real_data.drop(['UScpi_ratio'],axis=1,inplace=True)
    real_data['view_cot'] = view_cot
    #3/回归系数<选择120个数据滚动回归>，同时做因变量预测
    linreg = linear_model.LinearRegression()
    Reg_win = 120
    forcast_gold = pd.DataFrame()
    for i in range(len(real_data)-Reg_win):
        x_par = real_data.iloc[0:i+Reg_win,:].loc[:,['USrdi','USrty']]
        y_par = real_data.iloc[0:i+Reg_win,:].loc[:,['LD_sg']]
        model = linreg.fit(x_par,y_par)
        m = real_data.index[i+Reg_win-1].strftime('%Y-%m')
        next_m = next_month(m)
        x_test = real_data[next_m]
        y_pred = linreg.predict(x_test[['USrdi','USrty']])
        index_x = x_test.index
        y = pd.DataFrame(y_pred*x_test['UScpi'].values,index=index_x,columns=['gold_forcast'])
        y['intercept'] = linreg.intercept_ 
        y['coef_USrdi'] = np.ones(len(y_pred))*linreg.coef_.flatten()[0]
        y['coef_USrty'] = np.ones(len(y_pred))*linreg.coef_.flatten()[1]
        y['zscore'] = np.ones(len(y_pred))*model.score(x_par,y_par)
        result = pd.concat([y,x_test[['USrdi','USrty','UScpi']]],axis=1)
        forcast_gold = forcast_gold.append(result)

    forcast_gold = pd.concat([forcast_gold,real_data.loc[:,['LD_sg','view_cot']]],axis=1,join_axes=[forcast_gold.index])

    #4/结果
    view_gold,gold_smooth = cal_view(forcast_gold[['gold_forcast','USrdi','USrty']],para=0.34,win_obs=obs1)
    forcast_gold['view_gold'] = view_gold
    forcast_gold['smooth'] = gold_smooth
    if forcast_gold.index[-1].month == index_month[-1].month:
        now = datetime.now()
    else:
        now = index_month[-1]

    today_date = datetime(now.year,now.month,now.day)
    forcast_gold.loc[today_date] = None #增加一行
    forcast_gold['view'] = forcast_gold.loc[:,['view_gold']].values.flatten() * np.where(forcast_gold.view_gold.values*forcast_gold.view_cot.values <= 0,0,1)
    forcast_gold['view_01'] = np.where(forcast_gold.view > 0,1,np.where(forcast_gold.view==0,0,-1))
    forcast_gold[['view_gold','view_cot']] = forcast_gold[['view_gold','view_cot']].fillna(0)
    name1 = ['view_01','view','view_gold','view_cot','gold_forcast','smooth','LD_sg','UScpi','USrdi','USrty','coef_USrdi','coef_USrty','intercept','zscore']
    name2 = ['view_01','view','view_gold','view_cot','gold_forcast','smooth','coef_USrdi','coef_USrty','intercept','zscore']
    forcast_gold[name2] = forcast_gold[name2].shift(1)
    forcast_gold = forcast_gold[name1]
    forcast_gold['view_cot'] = forcast_gold['view_cot']*10.0
    forcast_gold['LD_sg'] = forcast_gold['LD_sg']*forcast_gold['UScpi']
    forcast_gold.to_csv('forcast_gold.csv')

    return forcast_gold

##############     主体函数END     #################################

@gt.command()
@click.option('--start-date','startdate',default='2003-01-01',help=u'start date to calc')
@click.option('--end-date','enddate',default=datetime.today().strftime('%Y-%m-%d'),help=u'start date to calc')
@click.option('--viewid','viewid',default='BL.000001',help=u'macro timing view id')
@click.pass_context
def gold_view_update(ctx,startdate,enddate,viewid):
    #function:"更新结果"
    mv = cal_gold_view()
    today = datetime.now().strftime('%Y-%m-%d')
    df = {}
    df['globalid'] = np.repeat(viewid,len(mv))
    df['bl_date'] = mv.index
    df['bl_view'] = mv.view_01.values
    df['bl_index_id'] = np.repeat('120000014',len(mv))
    df['created_at'] = np.repeat(today,len(mv))
    df['updated_at'] = np.repeat(today,len(mv))
    df_new = pd.DataFrame(df).set_index(['globalid','bl_date'])

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_bl_view',metadata,autoload = True)
    columns = [
        t.c.globalid,
        t.c.bl_date,
        t.c.bl_view,
        t.c.bl_index_id,
        t.c.created_at,
        t.c.updated_at,
    ]
    s = select(columns,(t.c.globalid==viewid))
    df_old = pd.read_sql(s,db,index_col=['globalid','bl_date'],parse_dates=['bl_date'])
    df_new = df_new[df_old.columns]#保持跟df_old列名同步
    database.batch(db,t,df_new,df_old,timestamp=False)
    print('########### id=BL.000009 #######保存到表格：asset/ra_bl_view  ####')
    print('########### gold view date:',mv.index[-1].strftime('%Y-%m-%d'),'#####value:',mv.view_01.values[-1])

if __name__ == '__main__':
    #initial_update()
    pass


