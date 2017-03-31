#coding=utf8


import MySQLdb
import pandas as pd


trade = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"trade",
    "charset": "utf8"
}


passport = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"passport",
    "charset": "utf8"
}

recommend = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"recommend",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**trade)
    conn.autocommit(True)

    sql = 'select ya_uid, ya_identity_no, created_at from yingmi_accounts'

    df = pd.read_sql(sql, conn, index_col = ['ya_uid'])

    df['ya_identity_no'] = df['ya_identity_no'].apply(lambda x: x[6 : 14])

    #print df
    df.to_csv('yingmi_account.csv', encoding = 'utf8')

    conn  = MySQLdb.connect(**passport)
    conn.autocommit(True)

    sql = 'select id, province, city, carrier, device_info, created_at from users'
    df = pd.read_sql(sql, conn, index_col = ['id'])

    df.to_csv('user.csv', encoding = 'utf8')
    #print df


    conn  = MySQLdb.connect(**trade)
    conn.autocommit(True)
    sql = 'select yt_uid, yt_portfolio_txn_id, yt_fund_code, yt_trade_type, yt_placed_amount, yt_placed_date from yingmi_trade_statuses'
    df = pd.read_sql(sql, conn, index_col = ['yt_uid'])
    #print df
    df.to_csv('yingmi_trade_status.csv', encoding = 'utf8')

    conn  = MySQLdb.connect(**recommend)
    conn.autocommit(True)
    sql = 'select uq_uid, uq_question_id, uq_question_selection, uq_answer, uq_start_time from user_questionnaire_answers where uq_answer_status'
    df = pd.read_sql(sql, conn, index_col = ['uq_uid'])

    df.to_csv('user_question_answer.csv', encoding = 'utf8')
    #print df
