#coding=utf8


import string
from datetime import datetime
from datetime import timedelta
import pandas as pd


f = open('./data/dates','r')
line = f.readline()
f.close()



dates = set()
vec = line.strip().split(',')
for d in vec:
    dates.add(d)



f = open('./data/index_weekly.csv','r')
lines = f.readlines()
f.close()



print lines[0].strip()



for i in range(1, len(lines)):
    line = lines[i].strip()
    vec = line.strip().replace('/','-').split(',')
    d = vec[0].strip()
    d = (datetime.strptime(d, '%Y-%m-%d') + timedelta(2)).strftime('%Y-%m-%d')
    if d in dates:
        print d.strip(), line[line.index(','): len(line)].strip()
        


'''
df = pd.read_csv('./data/index_weekly.csv', index_col = 'date', parse_dates = [0])
df_index = df.index
for i in range(0, len(df_index)):
    df.index[i] = df_index[i] + timedelta(2)

df = df[ df['date'] in dates]
print df
#for d in df.index:
#    delta_d = d + timedelta(2)
#    str = delta_d.strftime('%Y-%m-%d')
#    print str
'''


