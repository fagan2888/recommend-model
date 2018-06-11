from db import *
from RiskMgrGARCHHelper import *
from scipy import stats
from datetime import datetime


def calc_when_riskctrl_triggered(index_code, current):
    tdate = base_trade_dates.load_origin_index_trade_date(index_code)
    nav = database.load_nav_series(index_code, reindex=tdate)
    inc = np.log(nav).diff().fillna(0)*100
    inc.loc[pd.to_datetime(datetime.now())] = current
    inc2d = inc.rolling(2).sum().fillna(0)
    inc3d = inc.rolling(3).sum().fillna(0)
    inc5d = inc.rolling(5).sum().fillna(0)
    df = pd.DataFrame({'inc2d':inc2d, 'inc3d':inc3d, 'inc5d':inc5d})
    from Queue import Queue
    q = Queue()
    calc_var(len(df)-1, df, q)
    day, var2d, var3d, var5d = q.get()

    print "======================="
    print "current returns:"
    print "2d: %f, 3d: %f, 5d: %f" % (inc2d[-1], inc3d[-1], inc5d[-1])
    print "new riskmgr: var"
    print "2d: %f, 3d: %f, 5d: %f" % (var2d, var3d, var5d)

    sr_cnfdn = inc5d.rolling(window=252).apply(lambda x: stats.norm.ppf(0.01, x.mean(), x.std(ddof=1)))
    print "old 5d var: %f" % sr_cnfdn[-1]

if __name__ == '__main__':
    index_code = 'ERI000002'
    current = -1.98
    calc_when_riskctrl_triggered(index_code, current)

