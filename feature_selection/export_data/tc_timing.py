import pandas as pd
import MySQLdb

config = {
        'host': '127.0.0.1',
        'port': 3306,
        'user': 'yaojiahui',
        'passwd': '3el79KPN2BkoFeRkNo0',
        'db': 'asset_allocation',
        'charset': 'utf8'
}

def export_from_db(tc_id):
    conn = MySQLdb.connect(**config)
    sql = 'select tc_date, tc_signal from tc_timing_signal where tc_timing_id = \
            %d'%tc_id
    df = pd.read_sql(sql, conn, index_col = ['tc_date'], parse_dates = True)
    return df


def merge_data(asset, tc_id):
    asset_data = pd.read_csv('./assets/' + asset + '_indicator_day_data.csv', \
            index_col = 0, parse_dates = True)
    tc_signal_data = export_from_db(tc_id)
    tc_signal_data.index.name = 'date'
    pct_chg = asset_data['pct_chg'].loc[tc_signal_data.index]
    tc_signal_data['pct_chg'] = pct_chg
    tc_signal_data.dropna(inplace = True)
    tc_signal_data = tc_signal_data.loc[:, ['pct_chg', 'tc_signal']]
    return tc_signal_data

if __name__ == '__main__':
    #export_from_db(21110100)
    asset_tcid = {
            '120000001': 21110100,
            '120000002': 21110200,
            '120000013': 21120200,
            '120000014': 21400100,
            '120000015': 21120500,
            '120000029': 21400400,
}
    for asset, tc_id in asset_tcid.iteritems():
        merge_data(asset, tc_id).to_csv('./output_data/' + asset + '_td.csv')
