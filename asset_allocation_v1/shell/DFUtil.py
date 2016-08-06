#coding=utf8


import pandas as pd
import datetime
import calendar



def get_date_df(df, start_date, end_date):
    _df = df[df.index <= datetime.datetime.strptime(end_date,'%Y-%m-%d').date()]
    _df = _df[_df.index >= datetime.datetime.strptime(start_date,'%Y-%m-%d').date()]
    return _df



def last_friday():
    date   = datetime.date.today()
    oneday = datetime.timedelta(days=1)
    
    while date.weekday() != calendar.FRIDAY:
        date -= oneday

    date = date.strftime('%Y-%m-%d')
    return date
