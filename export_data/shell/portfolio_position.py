#coding=utf8

import string
import MySQLdb


risk_names = []
for i in range(0,11):
    name = 'risk' + str(i) + '_2015-09-11'    
    risk_names.append(name)


fund_ids = set()

records = []
conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')
cur = conn.cursor()

for name in risk_names:
    cur.execute("select pw_fund_id, pw_weight from portfolio_weights where pw_portfolio_name = '%s' " % (name))
    record = cur.fetchall()
    for item in record:
        fund_ids.add(item[0])
    records.append(record)

cur.close()
conn.commit()
conn.close()

fund_names = {}
conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='mofang', charset='utf8')
cur = conn.cursor()
for id in fund_ids:
    cur.execute("select fi_name from fund_infos where fi_globalid = '%d'" % id)    
    record = cur.fetchone()
    fund_names[id] = record[0]

cur.close()
conn.commit()
conn.close()

for i in range(0, len(records)):
    record = records[i]
    for item in record:
        print i, ',' ,fund_names[item[0]],',',item[1]        
