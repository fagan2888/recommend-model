#coding=utf8

import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    print df['risk1']
