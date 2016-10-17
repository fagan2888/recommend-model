#coding=utf8


import ast
import getopt
import os
import sys
import numpy as np
import pandas as pd
import string
import GeneralizationTrade
import MySQLdb
import config

from Const import datapath
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from dateutil.parser import parse

conn  = None

def categories_types():
    #
    # 输出配置数据
    #
    return {
        'largecap'        : '11', # 大盘
        'smallcap'        : '12', # 小盘
        'rise'            : '13', # 上涨
        'oscillation'     : '14', # 震荡
        'decline'         : '15', # 下跌
        'growth'          : '16', # 成长
        'value'           : '17', # 价值

        'ratebond'        : '21', # 利率债
        'creditbond'      : '22', # 信用债
        'convertiblebond' : '23', # 可转债

        'money'           : '31', # 货币

        'SP500.SPI'       : '41', # 标普
        'GLNC'            : '42', # 黄金
        'HSCI.HI'         : '43', # 恒生
    }


def portfolio_position():
    columns =['date', 'largecap', 'smallcap', 'rise', 'decline', 'growth', 'value']

    df_ratio_equalrisk = pd.read_csv(datapath('equalriskassetratio.csv'), index_col = 'date', parse_dates = ['date'], usecols=columns)
    df_ratio_highrisk = pd.read_csv(datapath('high_position.csv'), index_col = 'date', parse_dates = ['date'])
    df_ratio_lowrisk = pd.read_csv(datapath('low_position.csv'), index_col = 'date', parse_dates = ['date'])

    #
    # 基本思想:
    #
    #    时间轴 采用马克维茨调仓点和风险修型调仓点的并集, 起点采用三者起点中最大值
    #
    #    某中类资产的比例=(高|风险资产比例) * 该资产的马克维茨比例 * 该资产的风险修型仓位
    #
    # 其中, 如果是非股票类资产, 风险修型仓位始终未1. 例如:
    #    大盘股比例 = 高风险资产比例 * 大盘股的马克维茨比例 * 大盘资产的仓位
    #    利率债比例 = 低风险资产比例 * 利率债的马卡维茨比例
    #
    columns_high = df_ratio_highrisk.columns
    columns_low = df_ratio_lowrisk.columns
    
    start = max(df_ratio_highrisk.index.min(), df_ratio_lowrisk.index.min(), df_ratio_equalrisk.index.min())
    
    index = df_ratio_highrisk.index.union(df_ratio_lowrisk.index).union(df_ratio_equalrisk.index)
    df_ratio_highrisk = df_ratio_highrisk.reindex(index, method='pad')[start:]
    df_ratio_lowrisk = df_ratio_lowrisk.reindex(index, method='pad')[start:]
    df_ratio_equalrisk = df_ratio_equalrisk.reindex(index, method='pad')[start:]

    index = index[index >= start]
    
    # 构造马克维茨资产矩阵
    df_markowitz = pd.concat([df_ratio_highrisk, df_ratio_lowrisk], axis=1)
    # 扩展风险修型矩阵, 与马克维茨矩阵配型
    df_tmp1 = pd.DataFrame(1.0, index=index, columns=columns_high.difference(df_ratio_equalrisk.columns))
    df_tmp2 = pd.DataFrame(1.0, index=index, columns=df_ratio_lowrisk.columns)
    df_equalrisk = pd.concat([df_ratio_equalrisk, df_tmp1, df_tmp2], axis=1)

    # 中类资产配置比例
    dt = dict()
    for risk in range(1, 11):
        # 配置比例
        ratio_h  = (risk - 1) * 1.0 / 9
        ratio_l  = 1 - ratio_h
        # 生成高低风险配置矩阵并与马克维茨矩阵配型
        df_high_low = pd.concat([
            pd.DataFrame(ratio_h, index=df_ratio_highrisk.index, columns=df_ratio_highrisk.columns),
            pd.DataFrame(ratio_l, index=df_ratio_lowrisk.index, columns=df_ratio_lowrisk.columns),
        ], axis=1)
        # 单个风险配置结果
        df_risk_result = df_high_low * df_markowitz * df_equalrisk
        df_risk_result['money'] = 1 - df_risk_result.sum(axis=1)
        dt['%.1f' % (risk / 10.0)] = df_risk_result
        
    #
    # 保存中类资产配置比例
    # 
    df_result = pd.concat(dt, names=('risk', 'date'))
    df_result.to_csv(datapath('portfolio_position.csv'))
    #
    # 返回结果
    #
    return df_result

def portfolio_category():
    columns_stock =['date', 'largecap', 'smallcap', 'rise', 'decline', 'growth', 'value']
    columns_bond =['date', 'ratebond', 'creditbond']
    #
    # 计算各中类资产的配置比例, 确定调仓时间轴
    #
    df_position = portfolio_position()
    index = df_position.unstack(0).index

    #
    # 加载基金池,并根据调仓时间轴整形
    #
    df_fund_stock = pd.read_csv(datapath('stock_fund.csv'), index_col = 'date', parse_dates = ['date'], usecols=columns_stock)
    df_fund_stock = df_fund_stock.reindex(index, method='pad')
    
    df_fund_bond = pd.read_csv(datapath('bond_fund.csv'), index_col = 'date', parse_dates = ['date'], usecols=columns_bond)
    df_fund_bond = df_fund_bond.reindex(index, method='pad')

    df_fund_money = pd.DataFrame("[u'213009']", index=index, columns=['money'])

    glnc_pivot = datetime.strptime('2013-08-22', '%Y-%m-%d')
    sr_glnc = pd.Series("[u'000216']", index=index, name='GLNC')
    sr_glnc[index < glnc_pivot] = "[u'320013']"

    df_fund_other = pd.concat([
        pd.DataFrame("[u'096001']", index=index, columns=['SP500.SPI']),
        pd.DataFrame(sr_glnc)
    ], axis=1)

    #
    # 生成分类资产配置结果
    #
    df_fund_risk = pd.concat([df_fund_stock, df_fund_other, df_fund_bond, df_fund_money], axis=1)
    df_fund = pd.concat({'%.1f' % (risk / 10.0):df_fund_risk for risk in range(1, 11)}, names=('risk', 'date'))

    df_result = pd.concat({'ratio':df_position, 'xfund':df_fund}, axis=1, names=('xtype','category'))
    df_result = df_result.stack(1)

    #
    # 滤掉过小的份额配置
    #
    df_result = df_result[df_result['ratio'] >= 0.0001]
    df_result['xfund'] = df_result['xfund'].map(lambda s: ':'.join(ast.literal_eval(s)))
    df_result.to_csv(datapath('cposition.csv'))

    #
    # 生成用于交换的gposition
    #
    df_result.rename(index=categories_types()).to_csv(datapath('gposition.csv'))

    return df_result

def portfolio_simple():
    columns=('risk','date','fund','ratio')
    #
    # 中类资产比例和对应基金
    #
    df = portfolio_category()
    print "portfolio_category"

    df2 = pd.DataFrame(columns=columns)
    print len(df.index)
    i = 0
    for key,row in df.iterrows():
        i += 1
        # codes2 = filter_by_status(date, codes)
        (risk, date, _ph) = key
        ratio = row['ratio']
        funds = row['xfund'].split(':')
        data = [(risk, date, fund, ratio) for (fund, ratio) in split_category_ratio(ratio, funds)]

        df2 = df2.append(pd.DataFrame(data, columns=columns), ignore_index=True)
        if (i % 100) == 0:
            print risk, date
    print "category to fund finish"
    #
    # 根据基金代码合并配置比例
    #
    df_result = df2.groupby(['risk', 'date', 'fund']).sum()
    df_result.reset_index(inplace=True)
    print "sum funds"

    #
    # 过滤掉过小的份额配置
    #
    df_result = df_result[df_result['ratio'] >= 0.009999]
    print "fileout small"

    #
    # 过滤掉与上期换手率小于3%
    #
    df_result = filter_by_turnover_rate(df_result, 0.03)
    print "filter_by_turnover_rate"

    #
    # 某天持仓补足100%
    #
    df_result = pad_sum_to_one(df_result)
    print "pad_sum_to_one"
    
    #
    # 计算资产组合净值
    #
    df_result.to_csv(datapath('position-s.csv'), header=False)

    return df_result

def filter_by_turnover_rate(df, turnover_rate):
    df_result = pd.DataFrame(columns=['risk', 'date', 'fund', 'ratio'])
    for k0, v0 in df.groupby('risk'):
        df_tmp = filter_by_turnover_rate_per_risk(v0, turnover_rate)
        if not df_tmp.empty:
            df_result = pd.concat([df_result, df_tmp])
            
    return df_result

def filter_by_turnover_rate_per_risk(df, turnover_rate):
    df_result = pd.DataFrame(columns=['risk', 'date', 'fund', 'ratio'])
    df_last=None
    for k1, v1 in df.groupby(['risk', 'date']):
        if df_last is None:
            df_last = v1[['fund','ratio']].set_index('fund')
            df_result = pd.concat([df_result, v1])
        else:
            df_current = v1[['fund', 'ratio']].set_index('fund')
            df_diff = df_current - df_last
            xsum = df_diff['ratio'].sum()
            if df_diff.isnull().values.any() or abs(df_diff['ratio'].sum()) >= turnover_rate:
                df_result = pd.concat([df_result, v1])
                df_last = df_current
                
    return df_result

def pad_sum_to_one(df):
    df.reset_index(inplace=True)
    df3 = df['ratio'].groupby([df['risk'], df['date']]).agg(['sum', 'idxmax'])
    df4 = df.set_index(['risk', 'date'])

    df5 = df4.merge(df3, how='left', left_index=True, right_index=True)
    df5.ix[df3['idxmax'], 'ratio'] += (1 - df5.ix[df3['idxmax'], 'sum'])

    print df5['ratio'].groupby([df5.index.get_level_values(0), df5.index.get_level_values(1)]).sum()
    
    return df5[['fund', 'ratio']]

def split_category_ratio(ratio, funds):
    if ratio < 0.000099:
        return None
    #
    # 根据配置占比和可用基金数目平分基金
    #
    if ratio > 0.60 :
        count_used = min (5, len(funds))
    elif ratio > 0.45 :
        count_used = min (4, len(funds))
    elif ratio > 0.30 :
        count_used = min (3, len(funds))
    elif ratio > 0.15 :
        count_used = min (2, len(funds))
    else :
        count_used = 1;
        
    funds_used = funds[0:count_used]
    ratio_used = ratio / count_used

    return [(code, ratio_used) for code in funds_used]

def risk_position():
    fund_df                   = pd.read_csv(datapath('stock_fund.csv'), index_col = 'date', parse_dates = ['date'])
    bond_fund_df              = pd.read_csv(datapath('bond_fund.csv'), index_col = 'date', parse_dates = ['date'])
    equalrisk_ratio_df        = pd.read_csv(datapath('equalriskassetratio.csv'), index_col = 'date', parse_dates = ['date'])
    highriskposition_ratio_df = pd.read_csv(datapath('highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    lowriskposition_ratio_df  = pd.read_csv(datapath('lowriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    risk_portfolio_df         = pd.read_csv(datapath('risk_portfolio.csv') , index_col  = 'date', parse_dates = ['date'])
    label_asset_df            = pd.read_csv(datapath('labelasset.csv') , index_col  = 'date', parse_dates = ['date'])


    #print equalrisk_ratio_df
    #print highriskposition_ratio_df


    fund_df_index                = 0
    bond_fund_df_index           = 0
    equalrisk_ratio_index        = 0
    highriskposition_ratio_index = 0
    lowriskposition_ratio_index  = 0


    #risk_portfolio_index        = 0


    start_date = highriskposition_ratio_df.index[0]
    #print start_date
    equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index >= start_date]
    #
    dates = risk_portfolio_df.index
    start_date = dates[0]

    #print start_date
    #print dates[0]

    fund_codes       = {}
    bond_fund_codes  = {}
    equalrisk_ratio  = {}
    highriskposition = {}
    lowriskposition = {}

    #print fund_df
    for i in range(0, len(fund_df.index) - 1):
        if fund_df.index[i + 1] >= start_date:
            for j in range(0, len(fund_df.columns)):
                fund_codes[fund_df.columns[j]] = fund_df.iloc[i, j]
            fund_df = fund_df[fund_df.index > fund_df.index[i]]
            break

    #print fund_df
    for i in range(0, len(bond_fund_df.index) - 1):
        if bond_fund_df.index[i + 1] >= start_date:
            for j in range(0, len(bond_fund_df.columns)):
                bond_fund_codes[bond_fund_df.columns[j]] = bond_fund_df.iloc[i, j]
            bond_fund_df = bond_fund_df[bond_fund_df.index > bond_fund_df.index[i]]
            break


    for i in range(0, len(equalrisk_ratio_df.index) - 1):
        if equalrisk_ratio_df.index[i + 1] >= start_date:
            for j in range(0, len(equalrisk_ratio_df.columns)):
                equalrisk_ratio[equalrisk_ratio_df.columns[j]] = equalrisk_ratio_df.iloc[i, j]
            equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index > equalrisk_ratio_df.index[i]]
            break


    #print fund_codes

    #print fund_df
    #print equalrisk_ratio_df.index
    #print highriskposition_ratio_df.index
    #print risk_portfolio_df.index

    #print fund_df

    glnc_pivot = datetime.strptime('2013-08-22', '%Y-%m-%d')

    all_code_position = []

    for i in range(0, len(dates)):

        d = dates[i]

        if fund_df_index < len(fund_df.index) and d >= fund_df.index[fund_df_index]:
            for label in fund_df.columns:
                tmp_d = fund_df.index[fund_df_index]
                fund_codes[label] = fund_df.loc[tmp_d, label]
            fund_df_index = fund_df_index + 1


        if bond_fund_df_index < len(bond_fund_df.index) and d >= bond_fund_df.index[bond_fund_df_index]:
            for label in bond_fund_df.columns:
                tmp_d = bond_fund_df.index[bond_fund_df_index]
                bond_fund_codes[label] = bond_fund_df.loc[tmp_d, label]
            bond_fund_df_index = bond_fund_df_index + 1


        if equalrisk_ratio_index < len(equalrisk_ratio_df.index) and d >= equalrisk_ratio_df.index[equalrisk_ratio_index]:
            for label in equalrisk_ratio_df.columns:
                tmp_d = equalrisk_ratio_df.index[equalrisk_ratio_index]
                equalrisk_ratio[label] = equalrisk_ratio_df.loc[tmp_d, label]
            equalrisk_ratio_index = equalrisk_ratio_index + 1

        if highriskposition_ratio_index < len(highriskposition_ratio_df.index) and d >= highriskposition_ratio_df.index[highriskposition_ratio_index]:
            for label in highriskposition_ratio_df.columns:
                tmp_d = highriskposition_ratio_df.index[highriskposition_ratio_index]
                highriskposition[label] = highriskposition_ratio_df.loc[tmp_d, label]
            highriskposition_ratio_index = highriskposition_ratio_index + 1

        if lowriskposition_ratio_index < len(lowriskposition_ratio_df.index) and d >= lowriskposition_ratio_df.index[lowriskposition_ratio_index]:
            for label in lowriskposition_ratio_df.columns:
                tmp_d = lowriskposition_ratio_df.index[lowriskposition_ratio_index]
                lowriskposition[label] = lowriskposition_ratio_df.loc[tmp_d, label]
            lowriskposition_ratio_index = lowriskposition_ratio_index + 1


        #print d, fund_codes, equalrisk_ratio, highriskposition
        #print d,  bond_fund_codes, lowriskposition

        #print equalrisk_ratio
        #print d,



        for risk_rank in range(1, 11):
        #for risk_rank in range(10, 11):

            high_w  = (risk_rank - 1) * 1.0 / 9
            low_w = 1 - high_w

            #print highriskposition.keys()
            total_weight = 0
            ws = {}
            for col in highriskposition.keys():


                code = None

                if col == 'GLNC':
                    if d < glnc_pivot :
                        code = "[u'320013']"
                    else :
                        code = "[u'000216']"
                elif col == 'HSCI.HI':
                    code = "[u'000071']"
                elif col == 'SP500.SPI':
                    code = "[u'096001']"
                else:
                    code = fund_codes[col]

                highriskratio   = highriskposition[col]
                risk_ratio = equalrisk_ratio[col]
                #print d, code , highriskratio * risk_ratio * high_w
                #print risk_rank / 10.0, d, code , highriskratio * risk_ratio * high_w
                #all_code_position.append((risk_rank / 10.0, d, code ,highriskratio * risk_ratio * high_w))
                #print d, code, highriskratio * risk_ratio * high_w

                if col == 'GLNC':
                    risk_ratio = 1.0
                elif col == 'HSCI.HI':
                    risk_ratio = 1.0
                elif col == 'SP500.SPI':
                    risk_ratio = 1.0

                weight   = highriskratio * risk_ratio * high_w
                total_weight += weight
                all_code_position.append((d, risk_rank / 10.0, col, weight, code))

            for col in lowriskposition.keys():

                lowriskratio = lowriskposition[col]
                code         = bond_fund_codes[col]
                #print risk_rank / 10.0, d, code, lowriskratio * low_w
                #all_code_position.append((risk_rank / 10.0, d, code, lowriskratio * low_w))
                #print col,

                weight   = lowriskratio * low_w
                total_weight += weight
                all_code_position.append((d, risk_rank / 10.0, col, weight, code))

            if total_weight < 1 :
                code = "[u'213009']"
                left_weight = 1 - total_weight
                all_code_position.append((d, risk_rank / 10.0, 'money', left_weight, code))


    # all_code_position = clean_min(all_code_position)
    # all_code_position = clean_same(all_code_position)


    return all_code_position

def clean_same(re):
    tmp = {}
    key_list = {}
    for i in re:
        if tmp.has_key(str(i[0])+'--'+str(i[1])):
            tmp[str(i[0])+'--'+str(i[1])][str(i[2])] = i[3]
        else:
            tmp[str(i[0])+'--'+str(i[1])] = {}
            tmp[str(i[0])+'--'+str(i[1])][str(i[2])] = i[3]
        if key_list.has_key(str(i[0])):
            key_list[str(i[0])].append(str(i[1]))
        else:
            key_list[str(i[0])] = [str(i[1])]
    day_list = {}
    for k,v in key_list.items():
        t = {}
        for i in v:
            if t == {}:
                t = tmp[str(k)+'--'+str(i)]
                day_list[str(k)+'--'+str(i)]=1
            else:
                if is_same(tmp[str(k)+'--'+str(i)],t):
                    continue
                else:
                    t = tmp[str(k)+'--'+str(i)]
                    day_list[str(k)+'--'+str(i)]=1
    result = []
    for i in re:
        if day_list.has_key(str(i[0])+'--'+str(i[1])):
            result.append(i)
    return result


def is_same(tmp1,tmp2):
    flag = 0.03
    change = 0
    for k,v in tmp1.items():
        if tmp2.has_key(k):
            change += abs(v-tmp2[k])
        else:
            change += v
    for k,v in tmp2.items():
        if tmp1.has_key(k):
            pass
        else:
            change += v
    if change < flag:
        return True
    else:
        return False



def clean_min(re):
    tmp = {}
    tmp1 = {}
    tmp_list = []
    for i in re:
        if i[3] < 0.01:
            if tmp.has_key(str(i[0])+'--'+str(i[1])):
                tmp[str(i[0])+'--'+str(i[1])] += i[3]
            else:
                tmp[str(i[0])+'--'+str(i[1])] = i[3]
        else:
            if tmp1.has_key(str(i[0])+'--'+str(i[1])):
                tmp1[str(i[0])+'--'+str(i[1])] +=1
            else:
                tmp1[str(i[0])+'--'+str(i[1])] =1
            tmp_list.append(i)
    result = []
    kk = {}
    for i in tmp_list:
        c = list(i)
        if tmp.has_key(str(i[0])+'--'+str(i[1])):
            tmp1[str(i[0])+'--'+str(i[1])] -=1
            if tmp1[str(i[0])+'--'+str(i[1])] <= 0:
                if kk.has_key(str(i[0])+'--'+str(i[1])):
                    c[3] = round(i[3] + round(tmp[str(i[0])+'--'+str(i[1])]-kk[str(i[0])+'--'+str(i[1])],6)    ,6)
                else:
                    c[3] = round(i[3] * 1 / (1-tmp[str(i[0])+'--'+str(i[1])]) ,6)
            else:
                c[3] = round(i[3]* 1 / (1-tmp[str(i[0])+'--'+str(i[1])]) ,6)
                if kk.has_key(str(i[0])+'--'+str(i[1])):
                    kk[str(i[0])+'--'+str(i[1])] += round(c[3]-i[3],6)
                else:
                    kk[str(i[0])+'--'+str(i[1])] = round(c[3]-i[3],6)
        result.append(tuple(c))
    return result

def output_category_portfolio(all_code_position, out):
    #
    # 输出配置数据
    #
    xtab = {
        'largecap'        : 11, # 大盘
        'smallcap'        : 12, # 小盘
        'rise'            : 13, # 上涨
        'oscillation'     : 14, # 震荡
        'decline'         : 15, # 下跌
        'growth'          : 16, # 成长
        'value'           : 17, # 价值

        'ratebond'        : 21, # 利率债
        'creditbond'      : 22, # 信用债
        'convertiblebond' : 23, # 可转债

        'money'           : 31, # 货币

        'SP500.SPI'       : 41, # 标普
        'GLNC'            : 42, # 黄金
        'HSCI.HI'         : 43, # 恒生
    }

    for record in all_code_position:
        codes = ast.literal_eval(record[4])
        xtype = xtab[record[2]] if record[2] in xtab else 0
        out.write("%s,%.1f,%s,%.4f,%s\n" % (record[0].strftime("%Y-%m-%d"), record[1], xtype, record[3], ':'.join(codes)))

def output_portfolio(all_code_position, out):
    #
    # 输出配置数据
    #
    xtab = {
        'largecap'        : 11, # 大盘
        'smallcap'        : 12, # 小盘
        'rise'            : 13, # 上涨
        'oscillation'     : 14, # 震荡
        'decline'         : 15, # 下跌
        'growth'          : 16, # 成长
        'value'           : 17, # 价值

        'ratebond'        : 21, # 利率债
        'creditbond'      : 22, # 信用债
        'convertiblebond' : 23, # 可转债

        'money'           : 31, # 货币

        'SP500.SPI'       : 41, # 标普
        'GLNC'            : 42, # 黄金
        'HSCI.HI'         : 43, # 恒生
    }

    lines = []
    for record in all_code_position:
        codes = ast.literal_eval(record[4])
        xtype = xtab[record[2]] if record[2] in xtab else 0
        lines.append("%s,%.1f,%s,%.4f,%s" % (record[0].strftime("%Y-%m-%d"), record[1], xtype, record[3], ':'.join(codes)))

    #
    # 调用晓彬的代码
    #
    positions = GeneralizationTrade.init(lines)
    positions = clean_min(positions)
    positions = clean_same(positions)
    
    for record in positions :
        risk, date, code, ratio = record
        out.write("%s,%s,%06s,%.4f\n" % (risk, date, code, ratio))

        
def output_final_portfolio(all_code_position, out):
    global conn
    conn = MySQLdb.connect(**config.db_base)
    #
    # 输出配置数据
    #
    xtab = {
        'largecap'        : 11, # 大盘
        'smallcap'        : 12, # 小盘
        'rise'            : 13, # 上涨
        'oscillation'     : 14, # 震荡
        'decline'         : 15, # 下跌
        'growth'          : 16, # 成长
        'value'           : 17, # 价值

        'ratebond'        : 21, # 利率债
        'creditbond'      : 22, # 信用债
        'convertiblebond' : 23, # 可转债

        'money'           : 31, # 货币

        'SP500.SPI'       : 41, # 标普
        'GLNC'            : 42, # 黄金
        'HSCI.HI'         : 43, # 恒生
    }

    positions = []
    for record in all_code_position:
        date, risk, stype, ratio, codes, = record
        if ratio < 0.000099:
            continue
        codes = ast.literal_eval(codes)
        # xtype = xtab[stype] if stype in xtab else 0
        # print "%s,%.1f,%s,%.4f,%s" % (date.strftime("%Y-%m-%d"), risk, xtype, ratio, ':'.join(codes))
        #
        # 根据历史可购信息筛选基金
        #
        codes2 = filter_by_status(date, codes)
        #
        # 根据配置占比和可用基金数目平分基金
        #
        if ratio > 0.60 :
            count_used = min (5, len(codes2))
        elif ratio > 0.45 :
            count_used = min (4, len(codes2))
        elif ratio > 0.30 :
            count_used = min (3, len(codes2))
        elif ratio > 0.15 :
            count_used = min (2, len(codes2))
        else :
            count_used = 1;

        codes_used = codes2[0:count_used]
        ratio_used = ratio / count_used

        for code in codes_used :
            positions.append((risk, date, code, ratio_used))
            
        # if codes != codes2:
        #     print "diff", date.strftime("%Y-%m-%d"), codes, codes2
        
    # for record in positions:
    #     risk, date, code, ratio = record
    #     print "B:%.1f,%s,%06s,%.4f" % (risk, date.strftime("%Y-%m-%d"), code, ratio)

    positions = merge_same_fund(positions)
    positions = clean_min(positions)
    positions = clean_same(positions)

    for record in positions:
        risk, date, code, ratio = record
        out.write("%.1f,%s,%06s,%.4f\n" % (risk, date.strftime("%Y-%m-%d"), code, ratio))

    conn.close()

    # #
    # # 检查是否100%
    # #
    # positions = sorted(positions, key=itemgetter(0,1))
    # for key, group in groupby(positions, key = itemgetter(0,1)) :
    #     ratio = sum(e[3] for e in group)
    #     print "T:%.1f,%s,%.4f" % (key[0], key[1].strftime("%Y-%m-%d"), ratio)



#
# 合并相同基金比例. 先按照risk,date,fund分组, 对同一组的ratio求和
#
def merge_same_fund(positions) :
    result = []

    positions = sorted(positions, key=itemgetter(0,1,2))
    for key, group in groupby(positions, key = itemgetter(0,1,2)) :
        ratio = sum(e[3] for e in group)
        result.append(key + (ratio,))

    return result


#
# 根据历史可购信息筛选基金
#
def filter_by_status(date, codes):
    result = []
    # codes.extend([u'202202', u'000002'])

    imploded_codes = ','.join([repr(e.encode('utf-8')) for e in codes])


    min_date = parse('2016-09-03')
    if date <= min_date:
        # sql = "SELECT fs_fund_id, fs_date, fs_fund_code, fs_subscribe_status FROM fund_status WHERE fs_fund_code IN (%s) AND fs_date = '2016-09-03'" % (imploded_codes.decode('utf-8'))
        return codes
    else:
        sql = "SELECT fs_fund_id, fs_date, fs_fund_code, fs_subscribe_status FROM fund_status WHERE fs_fund_code IN (%s) AND fs_date = '%s'" % (imploded_codes.decode('utf-8'), date.strftime('%Y-%m-%d'))

    df = pd.read_sql(sql, con=conn, index_col='fs_fund_code', parse_dates=('fs_date'))

    df2 = df.reindex(codes, fill_value=0)
    
    result =  list(df2[df2.fs_subscribe_status == 0].index)
    return result


if __name__ == '__main__':

    # df2 = pd.read_csv('../testcases/aa.csv',  parse_dates=['date'])
    # print filter_by_turnover_rate(df2, 0.03)
    
    final = False
    category = False

    #
    # 处理命令行参数
    #
    try:
        longopts = ['datadir=', 'verbose', 'help', 'final', 'category']
        options, remainder = getopt.gnu_getopt(sys.argv[1:], 'hvd:', longopts)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in options:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-d', '--datadir'):
            datadir = arg
        elif opt in ('-v', '--verbose'):
            verbose = True
        elif opt == '--version':
            version = arg
        elif opt == '--final':
            final = True
        elif opt == '--category':
            category = True

    #
    # 确认数据目录存在
    #
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    else:
        if not os.path.isdir(datadir):
            print "path [%s] not dir" % datadir
            sys.exit(-1)

    #
    # 生成配置数据
    #
    all_code_position = risk_position()

    with (open(outfile, 'w') if outfile != '-' else os.fdopen(os.dup(sys.stdout.fileno()), 'w')) as out:
        if final:
            output_final_portfolio(all_code_position, out)
        else:
            if category :
                output_category_portfolio(all_code_position, out)
            else :
                output_portfolio(all_code_position, out)

