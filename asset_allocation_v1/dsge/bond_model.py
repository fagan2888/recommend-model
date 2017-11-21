import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression

def generate_data():
    bond = pd.read_csv('./data/bond.csv', index_col = 0, parse_dates = True)['bond']
    bond = bond.replace(0.0, np.nan).fillna(method = 'pad')
    bond = bond.resample('m').last()
    sf = pd.read_csv('./data/sf.csv', index_col = 0, parse_dates = True)['sf']
    m2_stock = pd.read_csv('./data/m2_stock.csv', index_col = 0, parse_dates = True)['m2_stock'].diff()

    df = pd.concat([bond, sf, m2_stock], 1).dropna().loc['2010':]
    df['sf_m2'] = df['sf'] - df['m2_stock']
    df['sf_m2_cumsum'] = df['sf_m2'].cumsum()
    regression = LinearRegression().fit(np.arange(len(df)).reshape(-1,1), df['sf_m2_cumsum'].values.reshape(-1,1))
    trend = regression.predict(np.arange(len(df)).reshape(-1,1))
    df['sf_m2_cumsum_trend'] = trend
    #df['sf_m2_cumsum_notrend'] = df['sf_m2_cumsum'] - df['sf_m2_cumsum_trend']
    df['sf_m2_cumsum_notrend'] = df['sf_m2_cumsum'].diff(12)
    #df.loc[:, 'sf_m2_cumsum'].plot()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(df['sf_m2_cumsum_notrend'], color = 'blue', label = 'sf-m2')
    ax2.plot(df['bond'], color = 'red', label = 'bond')
    ax1.legend(loc = 2)
    ax2.legend(loc = 1)
    plt.show()
    return df



if __name__ == '__main__':
    df = generate_data()
    print df
