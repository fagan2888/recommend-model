from __future__ import division
import pandas as pd

assets = ['120000001', '120000002','120000013','120000014','120000015',\
        '120000029']


def get_signal_wr(df, column):
    win_num = 0
    total_num = 0
    signal_nav = 1

    for i in range(len(df)-1):
        current = df[column][i]
        next_ = df[column][i+1]

        if current == next_:
            signal_nav *= (1 + df['pct_chg'][i+1]/100)
        elif current != next_:
            signal_nav *= (1 + df['pct_chg'][i+1]/100)
            if (signal_nav-1)*current >= 0:
                win_num += 1
            total_num += 1
            signal_nav = 1

    win_ratio = win_num/total_num
    print win_ratio
#    return win_ratio


for asset in assets:
    df_svm = pd.read_csv('./output_data/' + asset + '_weekly.csv', \
            index_col = 0, parse_dates = True)
    df_hmm = pd.read_csv('./output_data/' + asset + '_hmm.csv', \
            index_col = 0, parse_dates = True)
    df = (pd.concat([df_svm, df_hmm], axis = 1)).dropna()

    states = []
    state = -1
    for i in range(len(df)):
        if state == -1:
            if (df.means[i] > 0) and (df.pre_states[i] > 0):
                state = 1
        else:
            if (df.means[i] < 0) and (df.pre_states[i] < 0):
                state = -1
        states.append(state)
    df['states'] = states

    asset_nav = 1.0
    nav = 1.0
    hmm_nav = 1.0
    svm_nav = 1.0

    asset_nav_list = []
    nav_list = []
    hmm_nav_list = []
    svm_nav_list = []
    for i in range(len(states[:-1])):
        asset_nav *= (df.pct_chg[i+1]/100 + 1)
        asset_nav_list.append(asset_nav)
        if states[i] > 0:
            nav *= (df.pct_chg[i+1]/100 + 1)
            nav_list.append(nav)
        else:
            nav_list.append(nav)

        if df.means[i] > 0:
            hmm_nav *= (df.pct_chg[i+1]/100 + 1)
            hmm_nav_list.append(hmm_nav)
        else:
            hmm_nav_list.append(hmm_nav)

        if df.pre_states[i] > 0:
            svm_nav *= (df.pct_chg[i+1]/100 + 1)
            svm_nav_list.append(svm_nav)
        else:
            svm_nav_list.append(svm_nav)

    df = df[1:]
    df['asset_nav'] = asset_nav_list
    df['nav'] = nav_list
    df['hmm_nav'] = hmm_nav_list
    df['svm_nav'] = svm_nav_list
    #print df
    print 'svm signal wr:'
    get_signal_wr(df, 'pre_states')
    print 'hmm signal wr:'
    get_signal_wr(df, 'means')
    print 'combined signal wr:'
    get_signal_wr(df, 'states')
    #df.to_csv('./output_data/' + asset + '_contrast.csv')

    print 'asset: ', asset
    print 'asset_nav: ', asset_nav
    print 'nav: ', nav
    print 'hmm_nav: ', hmm_nav
    print 'svm_nav: ', svm_nav
    print

#print df
