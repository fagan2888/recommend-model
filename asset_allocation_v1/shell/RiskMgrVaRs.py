''' This module stands for calculating VaRs and signal of whether the joint distribution is brokethrough.
Because such a result should be written into DB, I just put all of the following code into a single file instead of 
being together with RiskMgrGARCH.
There are two classes: RiskMgrVaRs and RiskMgrJoint, they works similar.
'''


from RiskMgrGARCHHelper import *

import pandas as pd
from Queue import Queue


class RiskMgrVaRs(object):
    def __init__(self):
        pass

    def perform(self, nav):
        sr_inc = np.log(1+nav.pct_change().fillna(0))*100
        sr_inc2d = sr_inc.rolling(2).sum().fillna(0)
        sr_inc3d = sr_inc.rolling(3).sum().fillna(0)
        sr_inc5d = sr_inc.rolling(5).sum().fillna(0)
        df = pd.DataFrame({'inc2d': sr_inc2d,
                            'inc3d': sr_inc3d, 
                            'inc5d': sr_inc5d})
        df_vars = perform_single(df)
        return df_vars
    
    def perform_days(self, nav, idxs):
        sr_inc = np.log(1+nav.pct_change().fillna(0))*100
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
        

class RiskMgrJoint():
    def __init__(self):
        pass

    def perform(self, df_nav, target):
        joints = {}
        for global_id in df_nav.columns:
            if global_id != target:
                nav = df_nav.loc[:, [target, global_id]]
                #Here we first do `dropna` in case of not-aligned trade dates
                inc = np.log(1+nav.dropna().pct_change().fillna(0))*100
                inc5d = inc.rolling(5).sum().fillna(0)
                joints[global_id] = perform_joint(inc5d)
        df_result = pd.DataFrame(joints).fillna(False)
        return df_result

    def perform_days(self, df_nav, target, missing_days):
        joints = {}
        for global_id in df_nav.columns:
            if global_id != target:
                q = Queue()
                nav = df_nav.loc[:, [target, global_id]]
                inc = np.log(1+nav.dropna().pct_change().fillna(0))*100
                inc5d = inc.rolling(5).sum().fillna(0)
                idxs = [i for i in range(len(inc5d)) if inc5d.index[i] in missing_days]
                joint_status_days(idxs, inc5d, q)
                sr_constructor = {}
                while not q.empty():
                    day, signal, mu, cov = q.get()
                    sr_constructor[day] = signal
                joints[global_id] = pd.Series(sr_constructor)
        df_result = pd.DataFrame(joints).fillna(0)
        return df_result


