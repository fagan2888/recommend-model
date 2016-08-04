#coding=utf8

import string
import MySQLdb
import datetime

conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')

cur = conn.cursor()
last_date_sql = "select pf_date from portfolios order by pf_date desc limit 0, 1"
cur.execute(last_date_sql)
record = cur.fetchone()
last_date = record[0]
cur.close()
conn.commit()


portfolio_risk_id = {}
cur = conn.cursor()
portfolio_id_sql = "select id , p_risk from portfolios where pf_date = '%s'"
sql = portfolio_id_sql % (last_date)
cur.execute(sql)
for record in cur.fetchall():
    id = record[0]
    risk = record[1]
    portfolio_risk_id[risk] = id
cur.close()
conn.commit()


fund_ids = set()
cur = conn.cursor()
portfolio_weight_sql = "select pw_fund_id, pw_weight from portfolio_weights where pw_portfolio_id = '%s'"        
portfolio_risk_weight = {}
for risk in portfolio_risk_id.keys():
    id = portfolio_risk_id[risk]
    sql = portfolio_weight_sql % (id)
    weight = {}
    cur.execute(sql)
    for record in cur.fetchall():
        weight[record[0]] = record[1]        
        fund_ids.add(record[0])
    portfolio_risk_weight[risk] = weight

#print portfolio_risk_weight
#print fund_ids
cur.close()
conn.commit()



cur = conn.cursor()
cur.execute("select pv_date from portfolio_values order by pv_date desc limit 0,1")
record = cur.fetchone()
last_value_date = record[0]
#print last_value_date
cur.close()
conn.commit()


record = cur.fetchone()
portfolio_value = {}
portfolio_value_sql = "select pv_risk, pv_value from portfolio_values where pv_date = '%s'"
cur = conn.cursor()
cur.execute(portfolio_value_sql % (last_value_date))
for record in cur.fetchall():
    portfolio_value[record[0]] = record[1]
cur.close()
conn.commit()
conn.close()
#print portfolio_value


now = datetime.datetime.now()
#yesterday = (now + datetime.timedelta(days=-1))
yesterday = now

last_fund_values = {}
fund_values = {}

conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')

cur = conn.cursor()
for fid in fund_ids:
    
    sql = "select fv_authority_value from fund_value where fv_time = '%s' and fv_fund_id = '%d'" % (last_value_date, fid)
    cur.execute(sql)
    record =  cur.fetchone()
    value = record[0]
    i = -1
    while value == 0.0:
        day_str = (last_value_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        sql = "select fv_authority_value from fund_value where fv_time = '%s' and fv_fund_id = '%d'" % (day_str, fid)
        cur.execute(sql)
        record =  cur.fetchone()
        value = record[0]
        i = i - 1
    
    last_fund_values[fid] = value

cur.close()
conn.commit()

cur = conn.cursor()
for fid in fund_ids:
    
    sql = "select fv_authority_value from fund_value where fv_time = '%s' and fv_fund_id = '%d'" % (yesterday.strftime('%Y-%m-%d'), fid)
    print sql
    try:
        cur.execute(sql)
        record =  cur.fetchone()
        value = record[0]
        i = -1
        while value == 0.0:
            day_str = (yesterday + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            sql = "select fv_authority_value from fund_value where fv_time = '%s' and fv_fund_id = '%d'" % (day_str, fid)
            cur.execute(sql)
            record =  cur.fetchone()
            value = record[0]
            i = i - 1
        
        fund_values[fid] = value
    except:
        pass

cur.close()
conn.commit()
conn.close()

print fund_values.keys()
print fund_ids

if not (len(fund_values.keys()) == len(fund_ids)):
    exit(0)
    

conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')

cur = conn.cursor()
for risk in portfolio_risk_weight.keys():
    weight = portfolio_risk_weight[risk]
    p = 0.0
    for fid in weight.keys():
        p = p + (float)(weight[fid])* (fund_values[fid] / last_fund_values[fid] - 1)    

    #print risk
    #print portfolio_value[risk]    
    #print p
    v = (float)(portfolio_value[risk])  * (1 + p)
    #print v
    portfolio_value_sql = "insert into portfolio_values (pv_risk, pv_date, pv_value, pv_ratio, created_at, updated_at) values ('%f', '%s', '%f', '%f', '%s', '%s')" % (risk, yesterday.strftime('%Y-%m-%d'), v, p, datetime.datetime.now(), datetime.datetime.now())
    #cur.execute(portfolio_value_sql)    
    print portfolio_value_sql

cur.close()    
conn.commit()
conn.close()
    

