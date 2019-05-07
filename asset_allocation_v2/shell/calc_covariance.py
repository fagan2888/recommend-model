'''
Created on: May. 7, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''
import numpy as np
import pandas as pd
from datetime import datetime
import pymysql
import statsmodels.api as sm


# 计算协方差矩阵，时间加权
def yn_f_raw(data, half_life):  # type of data is np.array
    fun_data = data.copy()
    lambda_t = 0.5 ** (1 / half_life)
    weights_t = lambda_t ** (np.arange(fun_data.shape[0] - 1, -1, -1))
    weights_t = weights_t / weights_t.sum()
    f_raw = np.cov(fun_data, rowvar=False, aweights=weights_t, ddof=0)  # ddof=0: 样本方差 /n 默认为: ddof=1
    return f_raw


# 对上一步进行的协方差矩阵进行newey west 调整
def yn_newey_west(data, half_life, lags):
    fun_data = data.copy()
    lambda_t = 0.5 ** (1 / half_life)
    c_newey_west = np.zeros((fun_data.shape[1], fun_data.shape[1]))
    for i_lag in range(1, lags+1):
        c_newey_west_t = np.zeros((fun_data.shape[1], fun_data.shape[1]))
        for j_factor in range(fun_data.shape[1]):
            fun_data_t = fun_data.copy()  # Notice: use copy
            fun_data_j = fun_data_t[:-i_lag, j_factor]
            fun_data_t = fun_data_t[i_lag:, :]
            weights_t = lambda_t ** (np.arange(fun_data_t.shape[0] - 1, -1, -1))
            weights_t = weights_t / weights_t.sum()
            volatility_t = np.cov(fun_data_t, fun_data_j, rowvar=False, aweights=weights_t, ddof=0)
            c_newey_west_t[:, j_factor] = volatility_t[:-1, -1]
        coef_t = 1 - (i_lag / (lags + 1))
        c_newey_west = c_newey_west + coef_t * (c_newey_west_t + c_newey_west_t.T)
    return c_newey_west


# 对上一步计算的协方差矩阵进行特征值调整
def yn_eig_risk_adj(data, sample_period, M=1000, adj_coef=1.2):
    f_nw = data.copy()
    w_0, u_0 = np.linalg.eig(f_nw)
    d_0 = np.mat(u_0.T) * np.mat(f_nw) * np.mat(u_0)
    m_volatility = np.zeros((M, f_nw.shape[0]))
    # 模拟M次
    for m in range(M):
        b_m = np.zeros((f_nw.shape[0], sample_period))  # N*T
        for i_row in range(b_m.shape[0]):
            b_m[i_row, :] = np.random.normal(loc=0, scale=np.sqrt(d_0[i_row, i_row]), size=sample_period)
        r_m = np.mat(u_0) * np.mat(b_m)
        r_m = np.array(r_m.T)  # notice: 转置处理
        f_nw_m = np.cov(r_m, rowvar=False, ddof=0)  # 不需要Weights
        w_m, u_m = np.linalg.eig(f_nw_m)
        d_m = np.mat(u_m.T) * np.mat(f_nw_m) * np.mat(u_m)
        d_m_real = np.mat(u_m.T) * np.mat(f_nw) * np.mat(u_m)
        m_volatility[m, :] = np.diag(d_m_real) / np.diag(d_m)
    gamma_t = np.sqrt(m_volatility.mean(axis=0))
    gamma_t = adj_coef * (gamma_t - 1) + 1
    return gamma_t


def calc_covariance(data, lookback_period=512, H_L_vol=128, Lags_vol=32, H_L_corr=256, Lags_corr=32, Predict_period=10):
    fun_data = data.iloc[-lookback_period:].copy()  # type(data): dataframe
    fun_data_array = fun_data.copy().values
    # volatility
    F_raw_vol = yn_f_raw(data=fun_data_array, half_life=H_L_vol)
    C_Newey_West_vol = yn_newey_west(data=fun_data_array, half_life=H_L_vol, lags=Lags_vol)
    F_NW_vol = F_raw_vol + C_Newey_West_vol
    # correlation
    F_raw_corr = yn_f_raw(data=fun_data_array, half_life=H_L_corr)
    C_Newey_West_corr = yn_newey_west(data=fun_data_array, half_life=H_L_corr, lags=Lags_corr)
    F_NW_corr = F_raw_corr + C_Newey_West_corr
    # combine volatility and correlation
    F_NW = F_NW_corr.copy()
    for fun_i in range(F_NW.shape[0]):
        for fun_j in range(F_NW.shape[1]):
            F_NW[fun_i, fun_j] = F_NW[fun_i, fun_j] / np.sqrt(F_NW_corr[fun_i, fun_i] * F_NW_corr[fun_j, fun_j]) * np.sqrt(F_NW_vol[fun_i, fun_i] * F_NW_vol[fun_j, fun_j])
    F_NW = Predict_period * F_NW  # 日频方差--->月频方差
    # Eigenfactor Risk Adjustment
    gamma_array = yn_eig_risk_adj(data=F_NW, sample_period=fun_data.shape[0])
    gamma_matrix = np.mat(np.diag(gamma_array ** 2))
    W_0, U_0 = np.linalg.eig(F_NW)
    D_0 = np.mat(U_0.T) * np.mat(F_NW) * np.mat(U_0)
    D_real = gamma_matrix * D_0
    F_eigen = np.mat(U_0) * D_real * np.mat(U_0.T)
    return F_eigen

