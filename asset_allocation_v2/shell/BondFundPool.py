# -*- coding: UTF-8 -*-
import MySQLdb
import pandas as pd
import os
import datetime

db_base = {
        "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
        "port": 3306,
        "user": "koudai",
        "passwd": "Mofang123",
        "db": "caihui",
        "charset": "utf8"
}

db_mofang = {
        "host": "rdsijnrreijnrre.mysql.rds.aliyuncs.com",
        "port": 3306,
        "user": "koudai",
        "passwd": "Mofang123",
        "db":"mofang_api",
        "charset": "utf8"
}

conn  = MySQLdb.connect(**db_base)
conn_mofang = MySQLdb.connect(**db_mofang)

def load_save_fund_types():
    sql   = 'select SECURITYID, BEGINDATE, ENDDATE, L1NAME, L1CODES, L2NAME, L2CODES, L3NAME, L3CODES from TQ_FD_TYPECLASS'
    df    = pd.read_sql(sql, conn, index_col=['SECURITYID'])
    df.to_csv('../tmp/fund_types.csv', encoding='utf8')

    # 加载基金最低申购金额
    sql_fund_buy_amount = 'select fi_code, fi_amount, fi_yingmi_subscribe_status from fund_infos'
    df_fund_buy_amount = pd.read_sql(sql_fund_buy_amount, conn_mofang, index_col = 'fi_code')
    df_fund_buy_amount.to_csv("../tmp/bondfunds_buy_amount.csv", encoding="utf8")

def load_bond_funds(types):
    """
    加载types里指定分类代码的债券基金
    :param types: 债券基金分类代码
    :return: 返回相应代码基金,以dataframe格式返回
    """
    # load_save_fund_types()
    end_date_str = '19000101'
    end_date_int = 19000101
    end_date = datetime.datetime(1900, 1, 1)
    ftypes_df = pd.read_csv('../tmp/fund_types.csv')
    securityids = set()
    if len(types) != 0:
        for code in types:
            tmp_df = ftypes_df[(ftypes_df['L2CODES'] == code) & (ftypes_df['ENDDATE'] == end_date_int)]['SECURITYID']
            for sid in tmp_df.values:
                securityids.add(sid)
    else:
        tmp_df = ftypes_df['SECURITYID']
        for sid in tmp_df.values:
            securityids.add(sid)
    # 得到基金基本信息（财富）
    imploded_sids = ','.join([repr(sid) for sid in securityids])
    # print imploded_sids
    sql_base_info = 'select FDSNAME, MANAGERNAME, FSYMBOL, FOUNDDATE, ENDDATE, SECURITYID, KEEPERNAME from TQ_FD_BASICINFO where ISVALID = 1 and ENDDATE = ' + end_date_str + ' and OPERATEPERIOD is null and SECURITYID in (' + imploded_sids + ')'
    df_base_info  = pd.read_sql(sql_base_info, conn, index_col = 'SECURITYID', parse_dates = ['FOUNDDATE', 'ENDDATE'])

    # 得到基金份额
    sql_share = 'select SECURITYID, ENDDATE, TOTALSHARE from TQ_FD_FSHARE where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df_share = pd.read_sql(sql_share, conn, index_col = ['ENDDATE', 'SECURITYID'], parse_dates = ['ENDDATE'])
    df_share = df_share.unstack()
    df_share.columns = df_share.columns.droplevel(0)

    # 得到基金净值
    sql_nav = 'select SECURITYID, UNITNAV, NAVDATE from TQ_QT_FDNAV where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df_nav = pd.read_sql(sql_nav, conn, index_col = ['SECURITYID', 'NAVDATE'], parse_dates = ['NAVDATE'])

    # 基金经理
    sql_manager = 'select COMPCODE, SECURITYID, BEGINDATE, PSCODE, POST, ENDDATE, PSNAME, ISINCUMBENT, ENTRYDATE from TQ_FD_MANAGERINFO where ISVALID = 1 and POST = ' + '1003' + ' and SECURITYID in (' + imploded_sids + ')'
    df_manager = pd.read_sql(sql_manager, conn, index_col = ['COMPCODE','SECURITYID','BEGINDATE','PSCODE','POST'], parse_dates = ['BEGINDATE','ENDDATE', 'ENTRYDATE'])
    # 基金持有结构
    sql_share_holding = 'select SECURITYID, ENDDATE, INVTOTRTO from TQ_FD_SHARESTAT where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df_share_holding = pd.read_sql(sql_share_holding, conn, index_col = ['SECURITYID', 'ENDDATE'], parse_dates = ['ENDDATE'])

    # 基金资产组合
    sql_asset_port = 'select SECURITYID, REPORTDATE, SKRATIO from TQ_FD_ASSETPORTFOLIO where ISVALID = 1 and SECURITYID in (' + imploded_sids + ')'
    df_asset_port = pd.read_sql(sql_asset_port, conn, index_col = ['SECURITYID', 'REPORTDATE'], parse_dates = ['REPORTDATE'])
    #df_nav = df_nav.groupby(level = (0, 1)).first()
    #print df.loc['161211']
    #df.reset_index(inplace = True)
    #df = df.sort_values(by = ['FOUNDDATE'])
    #df = df.groupby(['FSYMBOL']).last()
    #df.set_index('SECURITYID')
    df_base_info.to_csv("../tmp/bondfunds_base_info.csv", encoding="utf8")
    df_share.to_csv("../tmp/bondfunds_share.csv", encoding="utf8")
    df_nav.to_csv("../tmp/bondfunds_nav.csv", encoding="utf8")
    df_manager.to_csv("../tmp/bondfunds_manager.csv", encoding="utf8")
    df_share_holding.to_csv("../tmp/bondfunds_share_holding.csv", encoding="utf8")
    df_asset_port.to_csv("../tmp/bondfunds_asset_port.csv", encoding="utf8")
    # df_reports.to_csv("../tmp/bondfunds_reports.csv", encoding="utf8")
    # delta = datetime.datetime(2016, 1, 1) - datetime.datetime(2015, 1, 1)
    # delta.days()
    #print df[:-1]

if __name__ == "__main__":
    #load_bond_funds([200204, 200301, 200302]) #, 200306])
    print("fund pool")
