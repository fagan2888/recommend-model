#coding=utf8


import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('./data/fsize.csv', index_col = 'date')
    dates = df.index
    d = dates[-1]
    sizes = df.loc[d].dropna()
    sizes.sort()
    print sizes
    sizes.to_csv('tmp.csv')

    #df = pd.read_csv('./data/fpe.csv', index_col = 'date')
    #dates = df.index
    #d = dates[-1]
    #pes = df.loc[d].dropna()
    #pes.sort()
    #print pes
