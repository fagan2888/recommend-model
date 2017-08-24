#coding=utf-8
import pandas as pd
import datetime

def dateindex(data):
    data = data.set_index(data.columns[0])
    try:
        data.index = map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'), data.index)
    except:
        try:
            data.index = map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'), data.index)
        except:
            data.index = map(lambda x: datetime.datetime.strptime(x,'%Y%m%d'), data.index)
    return data
