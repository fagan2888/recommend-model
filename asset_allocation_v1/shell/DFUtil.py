#coding=utf8

import pandas as pd
from datetime import datetime

def get_date_df(df, start_date, end_date):
	 _df = df[df.index <= datetime.strptime(end_date,'%Y-%m-%d').date()]
         _df = _df[_df.index >= datetime.strptime(start_date,'%Y-%m-%d').date()]
         return _df

