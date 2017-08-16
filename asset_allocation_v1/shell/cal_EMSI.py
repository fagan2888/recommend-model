import pandas as pd
from scipy.stats import spearmanr

def cal_emsi():
    close = pd.read_csv('~/recommend_model/asset_allocation_v1/sh300_component_close.csv', \
            index_col = 0, parse_dates = True)
    compo_stock = pd.read_csv('~/recommend_model/asset_allocation_v1/sh300_compo.csv')
    compo_stock = compo_stock.code.tolist()

    index = close.index
    ret = close.pct_change()
    std = close.rolling(5).std()
    emsi = []
    for i in range(len(index)):
        emsi.append(100*spearmanr(ret.loc[index[i]], std.loc[index[i]]).correlation)

    result_df = pd.DataFrame({'emsi':emsi}, index = index)
    result_df.to_csv('~/recommend_model/asset_allocation_v1/sh300_emsi.csv')

if __name__ == '__main__':
    cal_emsi()
