#input: high_df, low_df, assets_list: exclude other factors
def weighted(high_df, low_df,assets_list):
        
    for j in assets_list:
        high_df[j] = np.where(high_df[j] == 0.0, None, high_df[j])

    for j in assets_list:
        low_df[j]  = np.where(low_df[j] == 0.0, None, low_df[j])

    #high_df.to_excel('high.xlsx')
    #low_df.to_excel('low.xlsx')

    high_df['min'] = high_df.min(axis =1, skipna = True)
    high_df['max'] = high_df.max(axis =1, skipna = True)

    low_df['min'] = low_df.min(axis =1, skipna = True)
    low_df['max'] = low_df.max(axis =1, skipna = True)

    for j in assets_list:
        high_df[j] = ((high_df[j] - high_df['min']) / (high_df['max'] - high_df['min'])) + 1.0
        high_df[j] = np.where(high_df[j] == None, 0.0, high_df[j])

    for j in assets_list:
        low_df[j] = ((low_df[j] - low_df['min']) / (low_df['max'] - low_df['min'])) + 1.0
        low_df[j]  = np.where(low_df[j] == None, 0.0, low_df[j])

    del high_df['min'], high_df['max'], low_df['min'], low_df['max']

    high_df_w = high_df.abs()
    low_df_w  = 1.0 / low_df.abs()

    for j in assets_list:
        high_df_w[j] = np.where(high_df_w[j] == 0.0, None, high_df_w[j])

    for j in assets_list:
        low_df_w[j]  = np.where(low_df_w[j] == 0.0, None, low_df_w[j])
        low_df_w[j]  = np.where(low_df_w[j] == float('inf'), None, low_df_w[j])

    high_df_w.to_excel('high_df_w.xlsx')
    low_df_w.to_excel('low_df_w.xlsx')

    high_df_w['Q95'] = high_df_w.quantile(q=0.95, axis=1, numeric_only=False)
    high_df_w['Q05'] = high_df_w.quantile(q=0.05, axis=1, numeric_only=False)

    #Winsorized & weighted

    for j in assets_list:
        high_df_w[j] = np.where(high_df_w[j] > high_df_w['Q95'], high_df_w['Q95'], high_df_w[j])
        high_df_w[j] = np.where(high_df_w[j] < high_df_w['Q05'], high_df_w['Q05'], high_df_w[j])
    del high_df_w['Q95'], high_df_w['Q05']

    high_df_w['sum'] = high_df_w.sum(axis =1, skipna = True)
    for j in assets_list:
        high_df_w[j] = high_df_w[j]/high_df_w['sum']
    del high_df_w['sum']

    high_df_w = high_df_w.shift(1)

    low_df_w['Q95'] = low_df_w.quantile(q=0.95, axis=1, numeric_only=False)
    low_df_w['Q05'] = low_df_w.quantile(q=0.05, axis=1, numeric_only=False)

    for j in assets_list:
        low_df_w[j] = np.where(low_df_w[j] > low_df_w['Q95'], low_df_w['Q95'], low_df_w[j])
        low_df_w[j] = np.where(low_df_w[j] < low_df_w['Q05'], low_df_w['Q05'], low_df_w[j])
    del low_df_w['Q95'], low_df_w['Q05']

    low_df_w['sum'] = low_df_w.sum(axis =1, skipna = True)
    for j in assets_list:
        low_df_w[j] = low_df_w[j]/low_df_w['sum']
    del low_df_w['sum']

    low_df_w = low_df_w.shift(1)
    return high_df_w, low_df_w


#exp:
high_alpha_returns        = high_df_w * returns
high_alpha_returns['sum'] = high_alpha_returns.sum(axis =1, skipna = True)
high_alpha_returns['sum'] = high_alpha_returns['sum'].fillna(0)
high_alpha_returns['uv']  = 1.0
for i in range(1,len(high_alpha_returns.index)):
    high_alpha_returns['uv'][i] =  high_alpha_returns['uv'][i-1]*(1.0+high_alpha_returns['sum'][i])

high_alpha_returns.to_excel('high_alpha_returns.xlsx')

low_alpha_returns        = low_df_w * returns
low_alpha_returns['sum'] = low_alpha_returns.sum(axis =1, skipna = True)
low_alpha_returns['sum'] = low_alpha_returns['sum'].fillna(0)
low_alpha_returns['uv']  = 1.0
for i in range(1,len(low_alpha_returns.index)):
    low_alpha_returns['uv'][i] =  low_alpha_returns['uv'][i-1]*(1.0+low_alpha_returns['sum'][i])

low_alpha_returns.to_excel('low_alpha_returns.xlsx')
