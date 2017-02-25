#coding=utf8


import pandas as pd
import MySQLdb


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':

    other_fund_code_df = pd.read_csv('./data/otherfund_code.csv').dropna()
    codes = []
    for code in other_fund_code_df['code']:
        if code.find('OF') >= 0:
            codes.append(code[0:-3])

    conn  = MySQLdb.connect(**db_base)
    conn.autocommit(True)

    sql = 'select fi_code, fi_name, fi_subscribe_status, fi_redemption_status, fi_yingmi_amount, fi_yingmi_subscribe_status from fund_infos'

    df = pd.read_sql(sql, conn)
    df['fi_code'] = df['fi_code'].apply(lambda x : '%06d' % x)
    df = df.set_index('fi_code')

    df = df.loc[codes]
    df.to_csv('yingmi_buy.csv', encoding = 'utf8')
