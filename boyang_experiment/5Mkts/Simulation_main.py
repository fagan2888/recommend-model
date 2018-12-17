import scipy as sp
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
from collections import Counter
import statsmodels as sm
import random
import time
from datetime import timedelta, datetime
from multiprocess import Pool
# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from functools import partial

from sklearn import decomposition
#from ipdb import set_trace
from Index_Simulation_test import *


def Gaussian_Copula(data, Windows_size, suspension_tolerance_filtered_level, Nb_MC, Confidence_level, i):

    import numpy as np
    import pandas as pd
    import scipy as sp
    import time
    from datetime import datetime
    from Index_Simulation_test import Tranfer_simu2orig_data

    t = datetime.now()
    t = time.mktime(t.timetuple())
    seed = int(t+i)
    np.random.seed(seed)

    if Windows_size is not None:
        data = data.iloc[:Windows_size + i, :]
    # data_tolerance_filtered = filter_data_by_nan(data, suspension_tolerance_filtered_level)
    # data_tolerance_filtered = data_tolerance_filtered.fillna(method='bfill')
    data_tolerance_filtered = data.fillna(method='bfill')
    # 'Differernce between the log return and the arithmetic return'
    log_ret = np.log(data_tolerance_filtered /
                     data_tolerance_filtered.shift(1))

    log_ret = log_ret.iloc[1:, :]
    log_ret = log_ret.fillna(value=0)

    'Generate values from a multivariate normal distribution with specified mean vector and covariance matrix and the time is the same in histroy'

    cholesky_deco_corr_log_ret = np.linalg.cholesky(log_ret.corr())
    # Gaussian_Copula_Simulation = [(np.mean(log_ret) + np.dot(cholesky_deco_corr_log_ret, [
                                   # np.random.normal() for i in range(log_ret.shape[1])])).values.T for i in range(int(Nb_MC))]
    log_ret_mean = np.mean(log_ret)

    Nb_MC = int(Nb_MC)
    Gaussian_Copula_Simulation = [(log_ret_mean + np.dot(cholesky_deco_corr_log_ret, np.random.randn(log_ret.shape[1]))).values.T for i in range(Nb_MC)]
    # pool = Pool(32)
    # func = partial(gaussian_copula_simulate, log_ret_mean, cholesky_deco_corr_log_ret, log_ret.shape[1])
    # Gaussian_Copula_Simulation = pool.map(func, range(Nb_MC))
    # pool.close()
    # pool.join()

    Gaussian_Copula_Simulation = pd.DataFrame(data=np.stack(
        Gaussian_Copula_Simulation), columns=log_ret.columns)
    Gaussian_Copula_Simulation_cdf = pd.DataFrame(data=sp.stats.norm.cdf(
        Gaussian_Copula_Simulation), columns=log_ret.columns)

    Simulated_data_by_Gaussian_Copula = Tranfer_simu2orig_data(
        log_ret, Gaussian_Copula_Simulation_cdf)

    return Simulated_data_by_Gaussian_Copula


if __name__ == '__main__':

    Index_data = pd.read_csv("C:/Users/yshlm/Desktop/licaimofang/data/ra_index_nav_CN_US_HK.csv")
    columns_name = ['Index_Code', 'Trd_dt', 'Index_Cls']
    Index_data.columns = columns_name
    Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))

# USDCNY = pd.read_csv(
#    r"C:\Users\yshlm\Desktop\licaimofang\data\USDCNY.csv")
#USDCNY.columns = ['Trad_dt', 'PCUR', 'EXCUR', 'Cls']
#USDCNY.Trad_dt = USDCNY.Trad_dt.map(lambda x: pd.Timestamp(x))
#USDCNY.index = USDCNY.Trad_dt.map(lambda x: pd.Timestamp(x))

    'Expand the ret data to [0,1]'

    Index_data = Clean_data(Index_data)
    Index_data.index = Index_data.index.map(lambda x: pd.Timestamp(x))
    Index_data = Index_data.fillna(method='bfill')
    del Index_data['399006']

    Index_data_Chg = Index_data.pct_change()
    Index_data_Chg_Acc = Index_data_Chg.cumsum()
    Index_data_Chg_Acc = Index_data_Chg_Acc[
        Index_data_Chg_Acc.index > '2000-01-01']


    Copula = Gaussian_Copula(
        data=Index_data, Windows_size=None, suspension_tolerance_filtered_level=0.1, Nb_MC=1e+5, Confidence_level=0.99, i=1)

    q = 0.01
    Copula1 = Copula.quantile(q)

##########################################################################
    'What is the length of backtesting'
    '''
    ShortFall_Date_Summary = pd.DataFrame()
    for i in range(Copula1.shape[0]):

        ShortFall_Date = Index_data_Chg[Copula1.index[i]][
            Index_data_Chg[Copula1.index[i]].values <= Copula1[i]]
        ShortFall_Date1 = [Index_data_Chg.index[
            i] in ShortFall_Date for i in range(Index_data_Chg.shape[0])]
        ShortFall_Date1 = pd.DataFrame(data=ShortFall_Date1, columns=[
                                    Copula1.index[i]], index=Index_data_Chg.index)
        ShortFall_Date_Summary = pd.merge(
            ShortFall_Date_Summary, ShortFall_Date1, left_index=True, right_index=True, how='outer')


    ShortFall_Date_Summary_1D = pd.DataFrame(
        ShortFall_Date_Summary.values, columns=ShortFall_Date_Summary.columns)
    ShortFall_Date_Summary_1D.index = ShortFall_Date_Summary.index + \
        timedelta(days=1)


    ShortFall_Date_Summary_summary = Index_data_Chg[ShortFall_Date_Summary]
    ShortFall_Date_Summary_1D_summary = Index_data_Chg[ShortFall_Date_Summary_1D]


    print(ShortFall_Date_Summary_1D_summary.describe())


    ShortFall_Date_Summary_1D_summary_Acc = ShortFall_Date_Summary_1D_summary.fillna(
        value=0)
    ShortFall_Date_Summary_1D_summary_Acc = ShortFall_Date_Summary_1D_summary_Acc.cumsum()
    ShortFall_Date_Summary_1D_summary_Acc.plot(
        title='Accu Ret of shift 1-D trgger in %f' % (q))

    plt.show()
    '''
##########################################################################
    'Rolling or '

    Windows_size = 1000
    Windows_length = Index_data.shape[0] - Windows_size - 1

    # Timeseries_MC = [Gaussian_Copula(Index_data.iloc[:Windows_size + i, :], 0.1, 1e+4, 0.99)for i in range(Windows_length)]
    func = partial(Gaussian_Copula, Index_data, Windows_size, 0.1, 1e5, 0.99,)
    pool = Pool(32)
    Timeseries_MC = pool.map(func, range(Windows_length))
    pool.close()
    pool.join()

    Timeseries_VaR_Summary = np.stack(
        [Timeseries_MC[i].quantile(q).T for i in range(len(Timeseries_MC))])

#Timeseries_VaR=[Gaussian_Copula(Index_data.iloc[:Windows_size+i,:],0.1,1e+4,0.99).quantile(q) for i in range(Windows_length)]

    Timeseries_VaR_Summary = pd.DataFrame(data=Timeseries_VaR_Summary, columns=Copula.columns, index=Index_data.index[
                                        Windows_size:Windows_size + Windows_length])

    Index_backtesting = Index_data_Chg[
        Index_data_Chg.index.isin(Timeseries_VaR_Summary.index)]
    Index_backtesting_Indicator = Index_backtesting <= Timeseries_VaR_Summary
    Index_backtesting_Indicator_1D = pd.DataFrame(
        data=Index_backtesting_Indicator.values, columns=Index_backtesting_Indicator.columns, index=Index_backtesting_Indicator.index + timedelta(days=1))

    Index_data_Chg_Triggered = Index_data_Chg[
        Index_backtesting_Indicator_1D].fillna(value=0)
    Index_data_Chg_Triggered.to_csv('MC_result.csv')
# Index_data_Chg_Triggered1 = Index_data_Chg_Triggered.cumsum()

    'Stress testing for shock in different correlation area and in unexpected loss'
