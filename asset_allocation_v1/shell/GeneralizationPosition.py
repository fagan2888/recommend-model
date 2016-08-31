#coding=utf8


import ast
import getopt
import os
import sys
import pandas as pd
import string

from itertools import groupby
from operator import itemgetter
from Const import datadir

def risk_position():


    fund_df                   = pd.read_csv(os.path.join(datadir,'stock_fund.csv'), index_col = 'date', parse_dates = ['date'])
    bond_fund_df              = pd.read_csv(os.path.join(datadir,'bond_fund.csv'), index_col = 'date', parse_dates = ['date'])
    equalrisk_ratio_df        = pd.read_csv(os.path.join(datadir,'equalriskassetratio.csv'), index_col = 'date', parse_dates = ['date'])
    #highriskposition_ratio_df = pd.read_csv(os.path.join(datadir,'highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    highriskposition_ratio_df = pd.read_csv(os.path.join(datadir,'highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    lowriskposition_ratio_df  = pd.read_csv(os.path.join(datadir,'lowriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    risk_portfolio_df         = pd.read_csv(os.path.join(datadir,'risk_portfolio.csv') , index_col  = 'date', parse_dates = ['date'])
    label_asset_df            = pd.read_csv(os.path.join(datadir,'labelasset.csv') , index_col  = 'date', parse_dates = ['date'])


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
                all_code_position.append((d, risk_rank / 10.0, col, left_weight, code))
                

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
    print re
    print result
    for i in re:
        key = str(i[0])+'--'+str(i[1])
        if day_list.has_key(key):
            for code,ratio in tmp[key].items():
            	result.append((i[0],i[1],code,ratio)) 
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
        if i[3] < 0.01 and i[3]!=0:    
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

def output_category_portfolio(all_code_position):
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
        print "%s,%.1f,%s,%.4f,%s" % (record[0].strftime("%Y-%m-%d"), record[1], xtype, record[3], ':'.join(codes))

def output_final_portfolio(all_code_position):
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
        codes = ast.literal_eval(codes)
        xtype = xtab[stype] if stype in xtab else 0
        # print "%s,%.1f,%s,%.4f,%s" % (date.strftime("%Y-%m-%d"), risk, xtype, ratio, ':'.join(codes))
        if ratio > 0.60 :
            count_used = min (5, len(codes))
        elif ratio > 0.45 :
            count_used = min (4, len(codes))
        elif ratio > 0.30 :
            count_used = min (3, len(codes))
        elif ratio > 0.15 :
            count_used = min (2, len(codes))
        else :
            count_used = 1;

        codes_used = codes[0:count_used]
        ratio_used = ratio / count_used
        
        for code in codes_used :
            positions.append((risk, date, code, ratio_used))

    for record in positions:
        risk, date, code, ratio = record
        print "B:%.1f,%s,%06s,%.4f" % (risk, date.strftime("%Y-%m-%d"), code, ratio)

    positions = merge_same_fund(positions)       
    # positions = clean_min(positions) 
    # positions = clean_same(positions)

    for record in positions:
        risk, date, code, ratio = record
        print "A:%.1f,%s,%06s,%.4f" % (risk, date.strftime("%Y-%m-%d"), code, ratio)

def merge_same_fund(positions) :
    result = []
    
    positions = sorted(positions, key=itemgetter(0,1,2))
    for key, group in groupby(positions, key = itemgetter(0,1,2)) :
        ratio = sum(e[3] for e in group)
        result.append(key + (ratio,))

    return result


if __name__ == '__main__':

    final = False
    
    #
    # 处理命令行参数
    #
    try:
        longopts = ['datadir=', 'verbose', 'help', 'final']
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

    if final:
        output_final_portfolio(all_code_position)
    else:
        output_category_portfolio(all_code_position)

