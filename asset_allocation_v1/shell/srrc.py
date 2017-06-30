#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
#

import os
import MySQLdb
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import pandas as pd
import numpy as np
import datetime
from db import *
from config import *


conn = MySQLdb.connect(**db_portfolio_sta)
cur = conn.cursor(MySQLdb.cursors.DictCursor)
conn.autocommit(True)

sql ="SELECT date_format(ds_placed_date, '%Y-%m-01') AS rp_date, COUNT(DISTINCT ds_uid) as rp_user_newsub FROM `ds_order_pdate` WHERE `ds_trade_type` = 10 GROUP BY rp_date"
ds_user = pd.read_sql(sql, conn)
ds_user['rp_tag_id']=0

sql1="select date_format(ds_date,'%Y-%m-01') as rp_date,count(distinct ds_uid) as rp_user_hold from ds_share where ds_amount>0 and ds_date>'2016-08-01'group by rp_date"
#count=ds_order.groupby([ds_order['ds_trade_date'].apply(lamda x: x.month),ds_order['ds_trade_type'])['ds_uid'].count()

sql2="SELECT date_Format(ds_placed_date,'%Y-%m-01') as rp_date , count(DISTINCT ds_uid) as rp_user_resub , sum(ds_amount) as rp_amount_resub from ds_order_pdate WHERE ds_placed_date>'2016-08-01' and ds_trade_type=11 group by rp_date"

sql3="SELECT date_Format(ds_placed_date,'%Y-%m-01') as rp_date , count(DISTINCT ds_uid) as rp_user_clear from ds_order_pdate where ds_placed_date>'2016-08-01' and ds_trade_type in (30,31) group by rp_date"

sql4="SELECT date_Format(ds_placed_date,'%Y-%m-01') as rp_date , sum(ds_amount) as rp_amount_firstsub from ds_order_pdate where ds_placed_date>'2016-08-01' and ds_trade_type=10 group by rp_date"

sql5="SELECT date_Format(ds_placed_date,'%Y-%m-01') as rp_date , sum(ds_amount) as rp_amount_redeem from ds_order_pdate where ds_placed_date>'2016-08-01' and ds_trade_type in (20,21,30,31) group by rp_date"

sql6="select date_format(ds_date,'%Y-%m-01') as rp_date,sum(ds_amount) as rp_amount_aum from ds_share where ds_date>'2016-08-01' and ds_date in (select max_date from (select max(ds_date) as max_date from ds_share group by date_format(ds_date,'%Y-%m')) as aaa) group by rp_date"

hold=pd.read_sql(sql1,conn)
ds_user=pd.merge(ds_user,hold,how='outer',on=['rp_date'])

resub=pd.read_sql(sql2,conn)
ds_user=pd.merge(ds_user,resub,how='outer',on=['rp_date'])

clear=pd.read_sql(sql3,conn)
ds_user=pd.merge(ds_user,clear,how='outer',on=['rp_date'])

firstsub=pd.read_sql(sql4,conn)
ds_user=pd.merge(ds_user,firstsub,how='outer',on=['rp_date'])

redeem=pd.read_sql(sql5,conn)
ds_user=pd.merge(ds_user,redeem,how='outer',on=['rp_date'])

amount=pd.read_sql(sql6,conn)
ds_user=pd.merge(ds_user,amount,how='outer',on=['rp_date'])

df=ds_user.iloc[:,[2,0,1,4,6,3,7,5,8,9]]
created_dates = np.repeat(datetime.datetime.now(),len(df))
updated_dates = np.repeat(datetime.datetime.now(),len(df))
df['created_at']=created_dates
df['updated_at']=updated_dates
print df


df = df.set_index('rp_date')
print df

df.to_sql('rpt_srrc',conn,flavor='mysql',index=True,if_exists='append')


