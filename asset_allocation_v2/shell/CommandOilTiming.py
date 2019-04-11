#coding=utf-8

import sys
import click
import config
import logging
import trade_date
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ipdb import set_trace

from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData,Table,select

from db.asset_fundamental import *
from db import database,asset_trade_dates,base_ra_index_nav
from calendar import monthrange
from datetime import datetime,timedelta

from statsmodels.stats.outliers_influence import summary_table

#from sklearn import datasets,linear_model
# from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler,Normalizer,Binarizer,PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel#,MINE
from sklearn.ensemble import GradientBoostingClassifier,RandomForestRegressor,ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,RandomizedLasso,LogisticRegression,Ridge
#from sklearn.svm import LinearSVC

sys.path.append('shell/')
logger = logging.getLogger(__name__)


#############################################################################

@click.group(invoke_without_command=True)
@click.pass_context

def ot(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(oil_view_update)
    else:
        pass

################   数据读入    ##############################################

def load_oil_indicator():
    feature_names = {
        'MC.GD0018': 'USnrty',
        'MC.GD0032': 'brent_f_price',
        'MC.GD0033': 'wti_f_price',
        'MC.GD0034': 'oman_f_price',
        'MC.GD0035': 'wti_FOB',
        'MC.GD0036': 'brent_FOB',
        'MC.GD0037': 'dibai_FOB',
        'MC.GD0038': 'opec_price',
        'MC.GD0039': 'nymex_long_Fund',
        'MC.GD0040': 'nymex_short_Fund',
        'MC.GD0041': 'nymex_Arbitrage',
        'MC.GD0042': 'nymex_long_Commerce',
        'MC.GD0043': 'nymex_short_Commerce',
        'MC.GD0070': 'oilproduct_worldtotal',
        'MC.GD0071': 'oilopec_total',
        'MC.GD0095': 'oildemand_worldtotal',
        'MC.GD0201': 'oil_OPEC_Pro',
        'MC.GD0202': 'oil_OECD_consum',
        'MC.GD0203': 'oil_OECD_stocks',
        'MC.GD0204': 'oil_US_stocks',
        'MC.GD0205': 'oil_US_utiliza',
        'MC.GD0206': 'oil_US_Pro',
        'MC.GD0207': 'money_US_Index',
        'MC.GD0208': 'eco_US_GDP',
        'MC.GD0209': 'oil_US_export',
        'MC.GD0210': 'oil_world_demand',
        'MC.GD0211': 'oil_US_stocks2',
        'MC.GD0212': 'pos_ICE_longC',
        'MC.GD0213': 'pos_ICE_shortC',
        'MC.GD0214': 'pos_ICE_longF',
        'MC.GD0215': 'pos_ICE_shortF',
        'MC.GD0216': 'pos_Nymex_longC',
        'MC.GD0217': 'pos_Nymex_shortC',
        'MC.GD0218': 'pos_Nymex_longF',
        'MC.GD0219': 'pos_Nymex_shortF',
        'MC.GD0220': 'price_f_brent',
        'MC.GD0221': 'price_f_wti',
        'MC.GD0222': 'price_f_oman',
        'MC.GD0223': 'price_s_brent',
        'MC.GD0224': 'price_s_wti',
        'MC.GD0225': 'price_s_dibai',
        'MC.GD0050': 'EuroDollar',
        'MC.GD0051': 'GDP-A',
        'MC.GD0052': 'CPIr-A',
        'MC.GD0053': 'M2-A',
        'MC.GD0055': 'GDP-J',
        'MC.GD0056': 'M2-J',
        'MC.GD0058': 'CPIr-J',
        'MC.GD0059': 'J-ex',
        'MC.GD0060': 'GDP-G',
        'MC.GD0061': 'M2-G',
        'MC.GD0063': 'CPIr-G',
        'MC.GD0064': 'Euro-ex',
        'MC.GD0065': 'GDP-B',
        'MC.GD0066': 'M2-B',
        'MC.GD0068': 'CPIr-B',
        'MC.GD0069': 'B-ex',
        'MC.GD0072': 'GDP-I',
        'MC.GD0073': 'M2-I',
        'MC.GD0075': 'CPIr-I',
        'MC.GD0076': 'GDP-C',
        'MC.GD0077': 'M2-C',
        'MC.GD0079': 'CPIr-C',
        'MC.GD0080': 'C-ex',
        'MC.GD0081': 'GDP-R',
        'MC.GD0082': 'M2-R',
        'MC.GD0084': 'CPIr-R',
        'MC.GD0085': 'R-ex',
        'MC.GD0086': 'GDP-Ch',
        'MC.GD0087': 'M2-Ch',
        'MC.GD0088': 'CPIr-Ch',
        'MC.GD0089': 'Ch-ex',
        'MC.GD0090': 'Traded_DI',
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

#更新数据库函数
def oildata_save(gid,df):
    #读入旧数据
    db = database.connection('wind')
    t2 = Table('mc_gold_indicator',MetaData(bind=db),autoload=True)
    columns = [literal_column(c) for c in (df.index.names + list(df.columns))]
    s = select(columns).where(t2.c.globalid == gid)
    df_old = pd.read_sql(s,db,index_col=['globalid','mc_gold_date'],parse_dates=['mc_gold_date'])
    print ('##############################',df_old.tail())
    #更新数据库
    database.batch(db,t2,df,df_old,timestamp=False)


#更新数据库
def initial_update():
    feature_names = {
        'MC.GD0201': 'oil_OPEC_Pro',
        'MC.GD0202': 'oil_OECD_consum',
        'MC.GD0203': 'oil_OECD_stocks',
        'MC.GD0204': 'oil_US_stocks',
        'MC.GD0205': 'oil_US_utiliza',
        'MC.GD0206': 'oil_US_Pro',
        'MC.GD0207': 'money_US_Index',
        'MC.GD0208': 'eco_US_GDP',
        'MC.GD0209': 'oil_US_export',
        'MC.GD0210': 'oil_world_demand',
        'MC.GD0211': 'oil_US_stocks2',
        'MC.GD0212': 'pos_ICE_longC',
        'MC.GD0213': 'pos_ICE_shortC',
        'MC.GD0214': 'pos_ICE_longF',
        'MC.GD0215': 'pos_ICE_shortF',
        'MC.GD0216': 'pos_Nymex_longC',
        'MC.GD0217': 'pos_Nymex_shortC',
        'MC.GD0218': 'pos_Nymex_longF',
        'MC.GD0219': 'pos_Nymex_shortF',
        'MC.GD0220': 'price_f_brent',
        'MC.GD0221': 'price_f_wti',
        'MC.GD0222': 'price_f_oman',
        'MC.GD0223': 'price_s_brent',
        'MC.GD0224': 'price_s_wti',
        'MC.GD0225': 'price_s_dibai',
        'MC.GD0050': 'EuroDollar',
        'MC.GD0051': 'GDP-A',
        'MC.GD0052': 'CPIr-A',
        'MC.GD0053': 'M2-A',
        'MC.GD0055': 'GDP-J',
        'MC.GD0056': 'M2-J',
        'MC.GD0058': 'CPIr-J',
        'MC.GD0059': 'J-ex',
        'MC.GD0060': 'GDP-G',
        'MC.GD0061': 'M2-G',
        'MC.GD0063': 'CPIr-G',
        'MC.GD0064': 'Euro-ex',
        'MC.GD0065': 'GDP-B',
        'MC.GD0066': 'M2-B',
        'MC.GD0068': 'CPIr-B',
        'MC.GD0069': 'B-ex',
        'MC.GD0072': 'GDP-I',
        'MC.GD0073': 'M2-I',
        'MC.GD0075': 'CPIr-I',
        'MC.GD0076': 'GDP-C',
        'MC.GD0077': 'M2-C',
        'MC.GD0079': 'CPIr-C',
        'MC.GD0080': 'C-ex',
        'MC.GD0081': 'GDP-R',
        'MC.GD0082': 'M2-R',
        'MC.GD0084': 'CPIr-R',
        'MC.GD0085': 'R-ex',
        'MC.GD0086': 'GDP-Ch',
        'MC.GD0087': 'M2-Ch',
        'MC.GD0088': 'CPIr-Ch',
        'MC.GD0089': 'Ch-ex',
        'MC.GD0090': 'Traded_DI',
   }

    today = datetime.now().strftime('%Y-%m-%d')
    df = pd.read_csv('oildata_update1.csv')
    df = df.set_index(['date'])
    sorted_keys = sorted(feature_names.keys())
    print (sorted_keys)

    for i in range(0,len(sorted_keys)):
        key_i = sorted_keys[i]
        columns_name = feature_names[key_i]
        print (columns_name)
        df_i = df.loc[:,[columns_name]]
        df_i.index.name = 'mc_gold_date'
        df_i.columns = ['mc_gold_value']
        df_i.loc[:,'globalid'] = key_i
        df_i.loc[:,'created_at'] = today
        df_i.loc[:,'updated_at'] = today
        df_i = df_i.reset_index().set_index(['globalid','mc_gold_date'])
        print (key_i,df_i.tail(10))

        oildata_save(key_i,df_i)

###################   数据读入 -- END   #####################################

###################   辅助函数    ###########################################
def next_month(now_time):
    '''
    # 输入：字符串如'2018-06'
    # 输出：‘2018-7’
    '''
    next_year = int(now_time[0:4])
    next_month = int(now_time[5:7]) + 1
    if next_month == 13:
        next_month = 1
        next_year += 1
    month_time = str(next_year) + '-' + str(next_month)

    return month_time

def ema(s,n):
    #def/fucntion:"指数移动平均"
    ema = []
    j = 1
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1+n)
    ema.append(sma)
    ema.append(((s[n]-sma)*multiplier)+sma)

    for i in s[n+1:]:
        tmp = ((i - ema[j])*multiplier)+ema[j]
        j = j=1
        ema.append(tmp)
    ema1 = [s[:x].mean() for x in range(1,n)]
    ema = ema1+ema

    return np.array(ema)

def cal_cot(df,cot_rolling_win):
    #def/function:"计算cot值"
    if len(df) < cot_rolling_win:
        print ('Error')
    cal_df  =pd.DataFrame()
    cal_df['max'] = df.rolling(min_periods = 1,window=cot_rolling_win).max()
    cal_df['min'] = df.rolling(min_periods = 1,window=cot_rolling_win).min()
    cal_df['cot'] = (df - cal_df['min']) / (cal_df['max'] - cal_df['min'])

    return cal_df['cot'].fillna(0)

def cal_netcot(df,cot_win,ema_win):
    #def/function:"计算两cot值差"
    df_netcot = cal_cot(df.iloc[:,0],cot_win) - cal_cot(df.iloc[:,1],cot_win)
    cot = pd.DataFrame(ema(df_netcot.values, ema_win), index=df_netcot.index,columns=['netcot'])
    return cot

def filter_fuc(values,para):
    #def/function:简单过滤
    filter_values = list(values)
    for i in range(len(values)):
        if i> 2 and (values[i-1]*values[i-2] >= 0 and values[i-1]*values[i] <= 0 and abs(values[i]) < para):
            filter_values[i] = -1.0*values[i]
    return filter_values

def filter_fuc(values,para):
    #function:'简单过滤'
    filter_values = values
    for i in range(len(values)):
        if i > 2 and (values[i-1]*values[i-2] >= 0 and values[i-1]*values[i] <= 0 and abs(values[i]) < para):
            filter_values[i] = -1.0*values[i]
    return filter_values

def cal_view(df,para=0.34):
    #function:'计算观点值'
    df_data = df.iloc[:,0]
    strength = filter_fuc(pd.DataFrame(ema(df_data.values,12)).pct_change().fillna(0).values/para*100,para*5)
    return strength
##################    辅助函数  -- END   #####################################

##################    参数测试  --START  #####################################



#################     参数测试  --END   #####################################

#################     主体函数     ###########################################

def cal_oil_view(choice=3):
    #def/funciton:"计算原油的观点：基本面观点和资金面观点"
    #initial_update()
    data_origin = load_oil_indicator()

    ##################计算macro指标################
    data_names = ['GDP-A','M2-A','GDP-J','M2-J','GDP-G','M2-G','GDP-B','M2-B','GDP-I','M2-I','GDP-C','M2-C',\
        'GDP-R','M2-R','GDP-Ch','M2-Ch','J-ex','Euro-ex','B-ex','C-ex','R-ex','Ch-ex',\
        'CPIr-A','CPIr-J','CPIr-G','CPIr-B','CPIr-I','CPIr-C','CPIr-Ch']
    data = data_origin.loc[:,data_names]
    data = data.dropna().replace({0:np.nan}).fillna(method='ffill').resample('M').last()
    data = pd.concat([data,data_origin.loc[:,['CPIr-R']].replace({0:np.nan}).fillna(method='ffill').resample('M').last().pct_change(12)],axis=1).dropna()
    #美国数据：以万亿美元为单位，衡量GDP十亿和M2十亿
    data[['GDP-A','M2-A']] = data[['GDP-A','M2-A']]/1000.0
    #日本数据：日本GDP-十亿和M2亿
    data['GDP-J'] = data['GDP-J']/1000.0/data['J-ex']
    data['M2-J'] = data['M2-J']/1000.0/data['J-ex']
    #德国数据:GDP百万欧元，M2十亿欧元
    data['GDP-G'] = data['GDP-G']/1000000.0/data['Euro-ex']
    data['M2-G'] = data['M2-G']/1000.0/data['Euro-ex']
    #英国数据：GDP百万英镑，M2百万英镑
    data['GDP-B'] = data['GDP-B']/1000000.0/data['B-ex']
    data['M2-B'] = data['M2-B']/1000000.0/data['B-ex']
    #意大利数据：GDP百万英镑，M2百万英镑
    data['GDP-I'] = data['GDP-I']/1000000.0/data['Euro-ex']
    data['M2-I'] = data['M2-I']/1000000.0/data['Euro-ex']
    #加拿大数据：GDP百万加元，M2百万加元
    data['GDP-C'] = data['GDP-C']/1000000.0/data['C-ex']
    data['M2-C'] = data['M2-C']*data['GDP-C']#M2/GDP比例转化为M2
    ##俄罗斯数据：GDP十亿卢布，M2十亿卢布
    data['GDP-R'] = data['GDP-R']*4/1000.0/data['R-ex']
    data['M2-R'] = data['M2-R']/1000.0/data['R-ex']
    #中国数据：亿元
    data['GDP-Ch'] = data['GDP-Ch']*4/10000.0/data['Ch-ex']
    data['M2-Ch'] = data['M2-Ch']/10000.0/data['Ch-ex']
    #计算
    data['GDP'] = data[['GDP-A','GDP-J','GDP-G','GDP-B','GDP-I','GDP-C','GDP-R','GDP-Ch']].apply(lambda x:x.sum(),axis=1).values
    data['M2'] = data[['M2-A','M2-J','M2-G','M2-B','M2-I','M2-C','M2-R','M2-Ch']].apply(lambda x:x.sum(),axis=1).values
    data['M2_GDP'] = (data['M2']/data['GDP']).values - 1.0
    GDP_ratio = data[['GDP-A','GDP-J','GDP-G','GDP-B','GDP-I','GDP-C','GDP-R','GDP-Ch']].div(data['GDP'],axis='index')
    data['CPI'] = (data[['CPIr-A','CPIr-J','CPIr-G','CPIr-B','CPIr-I','CPIr-C','CPIr-R','CPIr-Ch']].values*GDP_ratio.values).sum(axis=1)
    data['unCPI'] = (data[['CPI']] - data[['CPI']].rolling(window=18,min_periods=1).mean()).values
    data_macro = data[['M2_GDP','CPI','unCPI']].shift(3).dropna()
    #print (data_macro.head(15))

    #################计算资金多空强度######################
    
    Pos_data = data_origin[['price_f_brent','price_f_wti','pos_ICE_longC','pos_ICE_shortC','pos_ICE_longF','pos_ICE_shortF','pos_Nymex_longC','pos_Nymex_shortC','pos_Nymex_longF','pos_Nymex_shortF']].truncate(before = '1995-03-21')
    Pos_data.iloc[0,2] = Pos_data['pos_ICE_longC'].loc[Pos_data['pos_ICE_longC'] > 0].head(1).values
    Pos_data.iloc[0,3] = Pos_data['pos_ICE_shortC'].loc[Pos_data['pos_ICE_shortC'] > 0].head(1).values
    Pos_data.iloc[0,4] = Pos_data['pos_ICE_longF'].loc[Pos_data['pos_ICE_longF'] > 0].head(1).values
    Pos_data.iloc[0,5] = Pos_data['pos_ICE_shortF'].loc[Pos_data['pos_ICE_shortF'] > 0].head(1).values
    Pos_data = Pos_data.replace(0,np.nan).fillna(method='ffill').resample('M').last().dropna()
    Pos_data['long'] = Pos_data[['pos_ICE_longC','pos_ICE_longF','pos_Nymex_longC','pos_Nymex_longF']].sum(axis=1).values
    Pos_data['short'] = Pos_data[['pos_ICE_shortC','pos_ICE_shortF','pos_Nymex_shortC','pos_Nymex_shortF']].sum(axis=1).values
    oil_netcot = cal_netcot(Pos_data[['long','short']],30,12).rename(columns={'netcot':'cot_T'})
    oil_netcot['price_f_wti'] = Pos_data['price_f_wti'].values
    #oil_netcot['price_forward'] = Pos_data['wti_f_price'].shift(1).values

#######################其他回归因素

    market_data = data_origin[['oil_OPEC_Pro','oil_OECD_consum','oil_OECD_stocks','oil_US_stocks','oil_US_Pro','money_US_Index','eco_US_GDP','USnrty']].truncate(before = '2000-12-31')
    market_data = market_data.dropna().replace(0,np.nan).fillna(method='ffill').resample('M').last().dropna()
    market_data[['oil_OPEC_Pro','oil_OECD_consum','oil_OECD_stocks','oil_US_stocks','oil_US_Pro']] = market_data[['oil_OPEC_Pro','oil_OECD_consum','oil_OECD_stocks','oil_US_stocks','oil_US_Pro']].pct_change(12).shift(1)
    market_data[['money_US_Index']] = market_data[['money_US_Index']].pct_change(1)
    market_data[['eco_US_GDP']] = market_data[['eco_US_GDP']].pct_change(12).shift(3)
    market_data = market_data.dropna()

######################回归
    oil_data = pd.concat([market_data,data_macro,oil_netcot],axis=1).dropna()
    name_par = ['oil_OPEC_Pro','oil_OECD_consum','oil_OECD_stocks','oil_US_stocks','oil_US_Pro','M2_GDP','CPI','unCPI','money_US_Index','USnrty','cot_T']
    x_par = oil_data.loc[:,name_par]
    y_par = oil_data.loc[:,['price_f_wti']]
    #x_par = pd.DataFrame(StandardScaler().fit_transform(x_par),columns=name_par,index=y_par.index)

    win_reg = 120
    rlasso_para = 0.94
    lr_para = 5
    y_pre = []
    for i in range(0,len(y_par)-1-win_reg):
        X = x_par.iloc[i:i+win_reg,:]
        Y = y_par.iloc[i:i+win_reg,:]
        names = name_par

        if choice==1:
            #特征选择算法1：稳定性选择<产生随机数，对于相同的输入可能会得到不同的结果>
            rlasso = RandomizedLasso(alpha=0.025)
            rlasso.fit(X,Y)
            rl_names = pd.DataFrame(sorted(zip(map(lambda x: round(x,4),rlasso.scores_),names),reverse=True),columns=['score','name'])
            rl_names = rl_names.loc[rl_names['score'] > rlasso_para]
            #rl_names = rl_names.iloc[0:4,:]
            select_names = rl_names['name'].values
            X = X[select_names]
            print ('Features sorted by their score:')
            print (select_names,rl_names['score'])
        if choice==2:
            #lsvc = LinearSVC(C=0.01,penalty='l1',dual=False).fit(X,Y)
            #X = SelectFromModel(lsvc,prefit=True).transform(X)
            clf = ExtraTreesClassifier().fit(X,Y)
            X = SelectFromModel(lcf,prefit=True).transform(X)
        if choice==3:
            #特征选择算法2：递归特征消除
            lr = Ridge(alpha=1.0)
            #lr = LinearRegression()
            rfe = RFE(lr, n_features_to_select=5)
            rfe.fit(X,Y)
            lr_names = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)),columns=['score','name'])
            lr_names = lr_names.loc[lr_names['score'] == 1]
            select_names = lr_names['name'].values
            X = X[select_names]
            print ("Features sorted by their rank:")
            print (select_names,lr_names['score'])

            #set_trace()
        #回归预测
        mod = sm.OLS(Y,X)
        res = mod.fit()
        y_pre.append(res.predict(x_par[select_names].iloc[i+win_reg,:].values)/res.predict(x_par[select_names].iloc[i+win_reg-1,:].values))

    df_y_pre = pd.DataFrame(y_pre,index=y_par.index[win_reg+1:],columns=['y_pre'])
    new_date = next_month(df_y_pre.index[-1].strftime('%Y-%m'))
    df_y_pre.loc[new_date] = None
    df_y_pre = pd.concat([df_y_pre.shift(1),y_par],axis=1)
    df_y_pre['y_pre'][0] = df_y_pre['price_f_wti'][0]
    df_y_pre = df_y_pre.loc[df_y_pre.y_pre > 0,:]
    df_y_pre[['y_pre']] = df_y_pre[['y_pre']].cumprod()
    df_y_pre['view_oil'] = cal_view(df_y_pre[['y_pre']])
    price_ema = ema(df_y_pre['price_f_wti'].values,12) - ema(df_y_pre['price_f_wti'].values,20)
    df_y_pre['view_technical']  = np.where(price_ema >= 0 ,np.where(price_ema > 0,1,0),-1)
    df_y_pre.to_csv('df_y_pre.csv')
    print (df_y_pre)


    return df_y_pre


if __name__ == '__main__':
    initial_update()
    cal_oil_view()

###################     主体函数 END     ######################################



@ot.command()
@click.option('--start-date','startdate',default='2003-01-01',help=u'start date to calc')
@click.option('--end-date','enddate',default=datetime.today().strftime('%Y-%m-%d'),help=u'start date to calc')
@click.option('--viewid','viewid',default='BL.000010',help=u'macro timing view id')
@click.pass_context

def oil_view_update(ctx,startdate,enddate,viewid):
    '''
    id=BL.000010,保存到表格：asset/ra_bl_view
    '''
    mv = cal_oil_view()
    today = datetime.now().strftime('%Y-%m-%d')
    union_mv = {}
    union_mv['globalid'] = np.repeat(viewid,len(mv))
    union_mv['bl_date'] = mv.index
    union_mv['bl_view'] = np.where(mv.view_oil >= 0,1,-1)
    union_mv['bl_index_id'] = np.repeat('120000014',len(mv))
    union_mv['created_at'] = np.repeat(today,len(mv))
    union_mv['updated_at'] = np.repeat(today,len(mv))
    union_mv_df = pd.DataFrame(union_mv,columns=['globalid','bl_date','bl_view','bl_index_id','created_at','updated_at'])
    df_new = union_mv_df.set_index(['globalid','bl_date'])

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

    s = select(columns,(t.c.globalid == viewid))
    df_old = pd.read_sql(s,db,index_col=['globalid','bl_date'],parse_date=['bl_date'])
    database.batch(db,t,df_new,df_old,timestamp=False)
    print ('######  id=BL.000010,保存到表格：asset/ra_bl_view ########')


