#coding=utf8

import MySQLdb
import sys
import string
import datetime
reload(sys)
sys.setdefaultencoding('utf8')


fund_values = {}
conn = MySQLdb.connect(host='182.92.214.1', port=3306, user='jiaoyang', passwd='Mofang123', db='mofang', charset='utf8')
cur = conn.cursor()
cur.execute("select fv_time, fv_authority_value from fund_value where fv_fund_id = 30000858")
for record in cur.fetchall():
	d = record[0]
	v = record[1]
	d_str = d.strftime("%Y-%m-%d")
	fund_values.setdefault(d_str, v)
cur.close()
conn.commit()
conn.close()


f = open('./data/hs_status','r')
hs_status = f.readlines()
hs_status = hs_status[1:len(hs_status)]
#print hs_status

f = open('./data/sharpe','r')
sharp = f.readlines()
sharp = sharp[1:len(sharp)]
sharp_ratio = {}
n = 1
for item in sharp:
	v = string.atof(item.strip().split(',')[1].strip())
	sharp_ratio.setdefault(n, v)	
	n = n + 1
sharp_ratio = sorted(sharp_ratio.iteritems(), key=lambda d:d[1], reverse = True)


tags = {}
(k, v) = sharp_ratio[0]
tags.setdefault(k, 'up')	
(k, v) = sharp_ratio[1]
tags.setdefault(k, 'mid')	
(k, v) = sharp_ratio[2]
tags.setdefault(k, 'down')	


f = open('./data/transition','r')
transition = f.readlines()
transition = transition[1:len(transition)]

current_status = string.atoi(hs_status[len(hs_status) - 1].strip().split(',')[1].strip())
prediction_s = 0;
p = -1;
vec =  transition[current_status - 1].strip().split(',')
for i in range(1, len(vec)):
	if string.atof(vec[i].strip()) > p:
		p = string.atof(vec[i].strip())
		prediction_s = i;

n = 0
i = 1
while True:
	s = string.atoi(hs_status[len(hs_status) - i].strip().split(',')[1].strip())	
	i = i + 1
	if s != current_status:
		n = i;
		break

start_date_str = hs_status[len(hs_status) - n].strip().split(',')[0].strip()
end_date_str = hs_status[len(hs_status) - 1].strip().split(',')[0].strip()
start_date_str = start_date_str[1:len(start_date_str) - 1]
end_date_str = end_date_str[1:len(end_date_str) - 1]



sql = "replace into hs300_statuses (date, value, status, created_at, updated_at) values ('%s',%f, '%s', '%s','%s')" 
history = []
for item in hs_status:
	item = item.strip()
	vec = item.strip().split(',')
	date = vec[0].strip()
	date = date[1:len(date) - 1]
	tag  = string.atoi(vec[1].strip())
	history.append((date , fund_values[date], tags[tag],  datetime.datetime.now(), datetime.datetime.now()))
	print sql % (date , fund_values[date], tags[tag],  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

prediction_sql = "insert into hs300_predictions (current_date_start, current_date_end, current_status, prediction_status,created_at, updated_at) values ('%s', '%s', '%s', '%s', '%s', '%s')"

conn = MySQLdb.connect(host='182.92.214.1', port=3306, user='jiaoyang', passwd='Mofang123', db='recommend', charset='utf8')
cur = conn.cursor()
for item in history:
	cur.execute(sql % item)
cur.close()
conn.commit()

cur = conn.cursor()
print prediction_sql % (start_date_str, end_date_str, tags[current_status], tags[prediction_s],datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
cur.execute(prediction_sql % (start_date_str, end_date_str, tags[current_status], tags[prediction_s],datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")));
cur.close()
conn.commit()
conn.close()

