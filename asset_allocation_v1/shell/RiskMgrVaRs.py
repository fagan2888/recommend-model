from RiskMgrGARCHHelper import *

import pandas as pd
from Queue import Queue


class RiskMgrVaRs(object):
    def __init__(self):
        pass

    def perform(self, nav):
        sr_inc = np.log(1+nav.dropna().pct_change()).fillna(0)*100
        sr_inc2d = sr_inc.rolling(2).sum().fillna(0)
        sr_inc3d = sr_inc.rolling(3).sum().fillna(0)
        sr_inc5d = sr_inc.rolling(5).sum().fillna(0)
        df = pd.DataFrame({'inc2d': sr_inc2d,
                            'inc3d': sr_inc3d, 
                            'inc5d': sr_inc5d})
        df_vars = perform_single(df)
        return df_vars
    
    def perform_days(self, nav, idxs):
        sr_inc = np.log(1+nav.dropna().pct_change()).fillna(0)*100
        sr_inc2d = sr_inc.rolling(2).sum().fillna(0)
        sr_inc3d = sr_inc.rolling(3).sum().fillna(0)
        sr_inc5d = sr_inc.rolling(5).sum().fillna(0)
        df = pd.DataFrame({'inc2d': sr_inc2d,
                            'inc3d': sr_inc3d, 
                            'inc5d': sr_inc5d})
        q = Queue()
        var2d_res = {}
        var3d_res = {}
        var5d_res = {}
        var_days(idxs, df, q)
        for i in range(q.qsize()):
            day, var2d, var3d, var5d = q.get()
            var2d_res[day] = var2d
            var3d_res[day] = var3d
            var5d_res[day] = var5d
        df_result = pd.DataFrame({'var_2d': var2d_res, 'var_3d': var3d_res, 'var_5d': var5d_res})
        return df_result
        

        
    
        

