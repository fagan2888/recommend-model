#coding=utf8


import getopt
import os
import sys
import pandas as pd
import TradeUtil
import Const

from Const import datapath

def risk_position():


    fund_df                   = pd.read_csv(datapath('stock_fund.csv'), index_col = 'date', parse_dates = ['date'])
    bond_fund_df              = pd.read_csv(datapath('bond_fund.csv'), index_col = 'date', parse_dates = ['date'])
    equalrisk_ratio_df        = pd.read_csv(datapath('equalriskassetratio.csv'), index_col = 'date', parse_dates = ['date'])
    #highriskposition_ratio_df = pd.read_csv(datapath('highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    highriskposition_ratio_df = pd.read_csv(datapath('highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    lowriskposition_ratio_df  = pd.read_csv(datapath('lowriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    risk_portfolio_df         = pd.read_csv(datapath('risk_portfolio.csv') , index_col  = 'date', parse_dates = ['date'])

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
    dates = risk_portfolio_df.index
    print "todiff:riskposition1", dates


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

    #print equalrisk_ratio
    #print fund_codes

    #print fund_df
    #print equalrisk_ratio_df.index
    #print highriskposition_ratio_df.index
    #print risk_portfolio_df.index

    #print fund_df

    all_code_position = []

    for i in range(0, len(dates)):

        d = dates[i]
        #print d

        if fund_df_index < len(fund_df.index) and d >= fund_df.index[fund_df_index]:
            for label in fund_df.columns:
                fund_codes[label] = fund_df.loc[d, label]
            fund_df_index = fund_df_index + 1


        if bond_fund_df_index < len(bond_fund_df.index) and d >= bond_fund_df.index[bond_fund_df_index]:
            for label in bond_fund_df.columns:
                bond_fund_codes[label] = bond_fund_df.loc[d, label]
            bond_fund_df_index = bond_fund_df_index + 1


        if equalrisk_ratio_index < len(equalrisk_ratio_df.index) and d >= equalrisk_ratio_df.index[equalrisk_ratio_index]:
            for label in equalrisk_ratio_df.columns:
                equalrisk_ratio[label] = equalrisk_ratio_df.loc[d, label]
            equalrisk_ratio_index = equalrisk_ratio_index + 1

        if highriskposition_ratio_index < len(highriskposition_ratio_df.index) and d >= highriskposition_ratio_df.index[highriskposition_ratio_index]:
            for label in highriskposition_ratio_df.columns:
                highriskposition[label] = highriskposition_ratio_df.loc[d, label]
            highriskposition_ratio_index = highriskposition_ratio_index + 1

        if lowriskposition_ratio_index < len(lowriskposition_ratio_df.index) and d >= lowriskposition_ratio_df.index[lowriskposition_ratio_index]:
            for label in lowriskposition_ratio_df.columns:
                lowriskposition[label] = lowriskposition_ratio_df.loc[d, label]
            lowriskposition_ratio_index = lowriskposition_ratio_index + 1


        #print d, fund_codes, equalrisk_ratio, highriskposition
        #print d,  bond_fund_codes, lowriskposition

        #print equalrisk_ratio
        #print d,



        for risk_rank in range(1, 11):

            high_w  = (risk_rank - 1) * 1.0 / 9
            low_w = 1 - high_w

            #print highriskposition.keys()
            ws = {}
            for col in highriskposition.keys():


                code = None

                if col == 'GLNC':
                    code = 216
                elif col == 'HSCI.HI':
                    code = 71
                elif col == 'SP500.SPI':
                    code = 96001
                else:
                    code = fund_codes[col]

                highriskratio   = highriskposition[col]
                risk_ratio = equalrisk_ratio[col]
                #print d, code , highriskratio * risk_ratio * high_w
                #print risk_rank / 10.0, d, code , highriskratio * risk_ratio * high_w
                #all_code_position.append((risk_rank / 10.0, d, code ,highriskratio * risk_ratio * high_w))
                #print d, code, highriskratio * risk_ratio * high_w
                weight   = ws.setdefault(code, 0.0)
                ws[code] = weight + highriskratio * risk_ratio * high_w

            for col in lowriskposition.keys():

                lowriskratio = lowriskposition[col]
                code         = bond_fund_codes[col]
                #print risk_rank / 10.0, d, code, lowriskratio * low_w
                #all_code_position.append((risk_rank / 10.0, d, code, lowriskratio * low_w))
                #print col,

                weight   = ws.setdefault(code, 0.0)
                ws[code] = weight  + lowriskratio * low_w

            for code in ws.keys():
                w = ws[code]
                all_code_position.append((risk_rank / 10.0, d, code, w))
        #print
    #print risk_portfolio_df
    #print fund_df

    #print all_code_position


    all_code_position = clean_min(all_code_position)
    all_code_position = clean_same(all_code_position)
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



if __name__ == '__main__':

    outfile = '-'
    
    #
    # 处理命令行参数
    #
    try:
        longopts = ['datadir=', 'verbose', 'help', 'output=']
        options, remainder = getopt.gnu_getopt(sys.argv[1:], 'hvd:o:', longopts)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in options:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-d', '--datadir'):
            Const.datadir = arg
        elif opt in ('-v', '--verbose'):
            verbose = True
        elif opt == '--version':
            version = arg
        elif opt in ('-o', '--output'):
            outfile = arg

    #
    # 确认数据目录存在
    #
    if not os.path.exists(Const.datadir):
        os.mkdir(Const.datadir)
    else:
        if not os.path.isdir(Const.datadir):
            print "path [%s] not dir" % Const.datadir
            sys.exit(-1)

    print Const.datadir
    all_code_position = risk_position()
    risk_dict = {}
    with (open(outfile, 'w') if outfile != '-' else os.fdopen(os.dup(sys.stdout.fileno()), 'w')) as out:
        for record in all_code_position:
            out.write("%.1f,%s,%06d,%.6f\n" % (record[0], record[1].strftime("%Y-%m-%d"), record[2], record[3]))
            
    #TradeUtil.getDailyNav(all_code_position)
    # print all_code_position
    # for tmp in all_code_position:
    #     if tmp[0] == 0.8:
    #         print str(tmp[1]) + "\t" +  str(tmp[2]) + "\t" +  str(tmp[3])

    '''
    fund_df                   = pd.read_csv(datapath('stock_fund.csv'), index_col = 'date', parse_dates = ['date'])
    bond_fund_df              = pd.read_csv(datapath('bond_fund.csv'), index_col = 'date', parse_dates = ['date'])
    equalrisk_ratio_df        = pd.read_csv(datapath('equalriskassetratio.csv'), index_col = 'date', parse_dates = ['date'])
    #highriskposition_ratio_df = pd.read_csv(datapath('highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    highriskposition_ratio_df = pd.read_csv(datapath('highriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    lowriskposition_ratio_df  = pd.read_csv(datapath('lowriskposition.csv'), index_col = 'date', parse_dates = ['date'])
    risk_portfolio_df         = pd.read_csv(datapath('risk_portfolio.csv') , index_col  = 'date', parse_dates = ['date'])

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
    dates = risk_portfolio_df.index


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

    #print equalrisk_ratio
    #print fund_codes

    #print fund_df
    #print equalrisk_ratio_df.index
    #print highriskposition_ratio_df.index
    #print risk_portfolio_df.index

    #print fund_df

    all_code_position = []

    for i in range(0, len(dates)):

        d = dates[i]
        #print d

        if fund_df_index < len(fund_df.index) and d >= fund_df.index[fund_df_index]:
            for label in fund_df.columns:
                fund_codes[label] = fund_df.loc[d, label]
            fund_df_index = fund_df_index + 1


        if bond_fund_df_index < len(bond_fund_df.index) and d >= bond_fund_df.index[bond_fund_df_index]:
            for label in bond_fund_df.columns:
                bond_fund_codes[label] = bond_fund_df.loc[d, label]
            bond_fund_df_index = bond_fund_df_index + 1


        if equalrisk_ratio_index < len(equalrisk_ratio_df.index) and d >= equalrisk_ratio_df.index[equalrisk_ratio_index]:
            for label in equalrisk_ratio_df.columns:
                equalrisk_ratio[label] = equalrisk_ratio_df.loc[d, label]
            equalrisk_ratio_index = equalrisk_ratio_index + 1

        if highriskposition_ratio_index < len(highriskposition_ratio_df.index) and d >= highriskposition_ratio_df.index[highriskposition_ratio_index]:
            for label in highriskposition_ratio_df.columns:
                highriskposition[label] = highriskposition_ratio_df.loc[d, label]
            highriskposition_ratio_index = highriskposition_ratio_index + 1

        if lowriskposition_ratio_index < len(lowriskposition_ratio_df.index) and d >= lowriskposition_ratio_df.index[lowriskposition_ratio_index]:
            for label in lowriskposition_ratio_df.columns:
                lowriskposition[label] = lowriskposition_ratio_df.loc[d, label]
            lowriskposition_ratio_index = lowriskposition_ratio_index + 1


        #print d, fund_codes, equalrisk_ratio, highriskposition
        #print d,  bond_fund_codes, lowriskposition

        #print equalrisk_ratio
        #print d,



        for risk_rank in range(1, 11):

            high_w  = (j - 1) * 1.0 / 9
            low_w = 1 - high_w

            for col in highriskposition.keys():

                #print col,

                code = None

                if col == 'GLNC':
                    code = 159937
                elif col == 'HSCI.HI':
                    code = 513600
                elif col == 'SP500.SPI':
                    code = 513500
                else:
                    code = fund_codes[col]

                highriskratio   = highriskposition[col]
                risk_ratio = equalrisk_ratio[col]
                #print d, code , highriskratio * risk_ratio * high_w
                #print risk_rank / 10.0, d, code , highriskratio * risk_ratio * high_w
                all_code_position.append((risk_rank / 10.0, d, code ,highriskratio * risk_ratio * high_w))

            for col in lowriskposition.keys():

                lowriskratio = lowriskposition[col]
                #print risk_rank / 10.0, d, code, lowriskratio * low_w
                all_code_position.append((risk_rank / 10.0, d, code, lowriskratio * low_w))
                #print col,


        #print
    #print risk_portfolio_df
    #print fund_df

    print all_code_position

    '''
