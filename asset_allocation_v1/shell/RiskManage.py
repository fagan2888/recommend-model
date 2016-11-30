# -*- coding: UTF-8 -*-
"""
Created at Nov 23, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
Company: LCMF
"""

import pandas as pd
import datetime
import numpy as np
import utils
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mtlab


class RiskManagement(object):

    def __init__(self):
        self.file_dir = "../tmp/"
        # 配置活跃幅
        self.port_pct = pd.read_csv(self.file_dir + "port_pct.csv",
                                    index_col=['date'], parse_dates=['date'])
        # 配置比例
        self.port_weight = pd.read_csv(self.file_dir + "port_weight.csv",
                                       index_col=['date'], parse_dates=['date'])
        # 沪深300择时信号
        self.tc_000300 = pd.read_csv(self.file_dir + "000300_gftd.csv",
                                     index_col=['date'], parse_dates=['date'])
        # 标普500择时信号
        self.tc_sp = pd.read_csv(self.file_dir + "sp_gftd.csv",
                                 index_col=['date'], parse_dates=['date'])
        # 恒生择时信号
        self.tc_hs = pd.read_csv(self.file_dir + "hs_gftd.csv",
                                 index_col=['date'], parse_dates=['date'])
        # 黄金择时信号
        self.tc_gold = pd.read_csv(self.file_dir + "gold_gftd.csv",
                                   index_col=['date'], parse_dates=['date'])
        # 需要风控的资产
        self.assets = ['sh000300', 'SP500.SPI', 'GLNC', 'HSCI.HI']
        # 卖出置信区间
        self.sell_base_ratio = 0.97
        # 买入置信区间
        self.buy_base_ratio = 0.75

    def risk_control(self):
        """
        风控核心逻辑
        :return: None
        """
        # 风控时间以修型时间为准
        dates = self.port_weight.index
        # 非沪深的另类市场资产时间，方便当另类资产时间与沪深不一样时取另类资产上一个日期
        dates_sp = self.tc_sp.index
        dates_hs = self.tc_hs.index
        dates_gold = self.tc_gold.index

        # 所有资产权重原始数据
        weights_origin = self.port_weight.copy()
        # 所有资产日涨跌幅数据（修型后的）
        pct_origin = self.port_pct.copy()

        # 风控开始时间（非风控开始作用时间，因为要回撤1年）
        start_time = dates[0]
        # 风控开始时间往后推一年的时间点
        next_year = start_time + datetime.timedelta(days=365*1.5)
        next_year = datetime.datetime.strptime(next_year.strftime("%Y-%m-%d"), "%Y-%m-%d")
        # 风控开始作用时间
        test_start = utils.get_move_day(dates, next_year, 1)
        # 风控结束时间
        test_end = dates[-1]
        # 风控开始作用时间（不断向后移动）
        inter_date = test_start

        # 各资产在配置中的比例，取风控开始时间到风控作用时间内的数据
        # （不要最后一个是因为test_start点为风控作用点，下面会在这个点append一个风控后权重）
        sh_weight = list(weights_origin[start_time:test_start]['sh000300'].values)[:-1]
        sp_weight = list(weights_origin[start_time:test_start]['SP500.SPI'].values)[:-1]
        hs_weight = list(weights_origin[start_time:test_start]['HSCI.HI'].values)[:-1]
        gold_weight = list(weights_origin[start_time:test_start]['GLNC'].values)[:-1]
        # 各资产在配置中的比例（所有数据）
        sh_weight_ori = np.array(weights_origin[start_time:test_end]['sh000300'].values)
        sp_weight_ori = np.array(weights_origin[start_time:test_end]['SP500.SPI'].values)
        hs_weight_ori = np.array(weights_origin[start_time:test_end]['HSCI.HI'].values)
        gold_weight_ori = np.array(weights_origin[start_time:test_end]['GLNC'].values)
        # 所有资产涨跌幅
        sh_pct = np.array(pct_origin[start_time:test_end]['sh000300'].values)
        sp_pct = np.array(pct_origin[start_time:test_end]['SP500.SPI'].values)
        hs_pct = np.array(pct_origin[start_time:test_end]['HSCI.HI'].values)
        gold_pct = np.array(pct_origin[start_time:test_end]['GLNC'].values)

        # 计算所有资产最大回撤，方便之后做分布
        [nav_sh, maxdown_sh] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['sh000300'].values))
        [nav_sp, maxdown_sp] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['SP500.SPI'].values))
        [nav_hs, maxdown_hs] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['HSCI.HI'].values))
        [nav_gold, maxdown_gold] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['GLNC'].values))
        del nav_sh, nav_sp, nav_hs, nav_gold

        maxdown_sh_ori = pd.DataFrame({"maxdown":maxdown_sh}, index=dates)
        maxdown_sp_ori = pd.DataFrame({"maxdown":maxdown_sp}, index=dates)
        maxdown_hs_ori = pd.DataFrame({"maxdown":maxdown_hs}, index=dates)
        maxdown_gold_ori = pd.DataFrame({"maxdown":maxdown_gold}, index=dates)

        back_date = len(sh_weight)
        # 是否按比例配置资产（风控作用之前的时间都按修型结果配置）
        sh_ttypes = np.ones(len(sh_weight))
        sp_ttypes = np.ones(len(sp_weight))
        hs_ttypes = np.ones(len(hs_weight))
        gold_ttypes = np.ones(len(gold_weight))

        # 0.97置信区间风控是否起作用
        sh_risk = False
        sp_risk = False
        hs_risk = False
        gold_risk = False

        # 风控起作用后的空仓时间
        sh_risk_days = 0
        sp_risk_days = 0
        hs_risk_days = 0
        gold_risk_days = 0
        # 择时是否起作用
        sh_signal_effect = False
        sp_signal_effect = False
        hs_signal_effect = False
        gold_signal_effect = False

        # 0.97在所有资产中起作次数
        risk_times = 0
        # 风控信号记录
        #沪深300 0.95起作用
        sh_95 = list(np.zeros(back_date))
        # 沪深300 0.75起作用
        sh_75 = list(np.zeros(back_date))
        # 沪深300 择时信号起作用
        sh_signal = list(np.zeros(back_date))
        # 沪深300 1.25起作用
        sh_125 = list(np.zeros(back_date))

        sp_95 = list(np.zeros(back_date))
        sp_75 = list(np.zeros(back_date))
        sp_signal = list(np.zeros(back_date))
        sp_125 = list(np.zeros(back_date))

        hs_95 = list(np.zeros(back_date))
        hs_75 = list(np.zeros(back_date))
        hs_signal = list(np.zeros(back_date))
        hs_125 = list(np.zeros(back_date))

        gold_95 = list(np.zeros(back_date))
        gold_75 = list(np.zeros(back_date))
        gold_signal = list(np.zeros(back_date))
        gold_125 = list(np.zeros(back_date))

        total_95 = list(np.zeros(back_date))
        total_75 = list(np.zeros(back_date))
        total_signal = list(np.zeros(back_date))
        total_125 = list(np.zeros(back_date))

        # 每天风控
        while inter_date <= test_end:

            if inter_date in dates:
                maxdown_sh = maxdown_sh_ori[maxdown_sh_ori.index.get_level_values(0) <= inter_date]['maxdown']
                maxdown_sp = maxdown_sp_ori[maxdown_sp_ori.index.get_level_values(0) <= inter_date]['maxdown']
                maxdown_hs = maxdown_hs_ori[maxdown_hs_ori.index.get_level_values(0) <= inter_date]['maxdown']
                maxdown_gold = maxdown_gold_ori[maxdown_gold_ori.index.get_level_values(0) <= inter_date]['maxdown']

                sh_pct_tmp = np.array(pct_origin[start_time:inter_date]['sh000300'].values)
                sp_pct_tmp = np.array(pct_origin[start_time:inter_date]['SP500.SPI'].values)
                hs_pct_tmp = np.array(pct_origin[start_time:inter_date]['HSCI.HI'].values)
                gold_pct_tmp = np.array(pct_origin[start_time:inter_date]['GLNC'].values)

                # 取过去一年各资产涨跌幅
                sh_pct_tmp = np.append([0.0], sh_pct_tmp[-5*54:])
                sp_pct_tmp = np.append([0.0], sp_pct_tmp[-5*54:])
                hs_pct_tmp = np.append([0.0], hs_pct_tmp[-5*54:])
                gold_pct_tmp = np.append([0.0], gold_pct_tmp[-5*54:])

                # 计算过去一年净值
                [nav_sh, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(sh_pct_tmp))
                [nav_sp, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(sp_pct_tmp))
                [nav_hs, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(hs_pct_tmp))
                [nav_gold, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(gold_pct_tmp))
                del maxdown_tmp
                # 每5个交易日对净值采样
                nav_sh = nav_sh[::5]
                nav_sp = nav_sp[::5]
                nav_hs = nav_hs[::5]
                nav_gold = nav_gold[::5]

                # 周涨跌幅
                pct_sh_week = np.diff(nav_sh)
                pct_sp_week = np.diff(nav_sp)
                pct_hs_week = np.diff(nav_hs)
                pct_gold_week = np.diff(nav_gold)
                # np.diff会去掉第一个数据（因为第一个数据没有涨跌幅），下面添加进去
                pct_sh_week = np.append([0.0], pct_sh_week)
                pct_sp_week = np.append([0.0], pct_sp_week)
                pct_hs_week = np.append([0.0], pct_hs_week)
                pct_gold_week = np.append([0.0], pct_gold_week)

                # 计算周回撤
                [nav_sh_week, maxdown_sh_week] = utils.cal_nav_maxdrawdown(list(pct_sh_week))
                [nav_sp_week, maxdown_sp_week] = utils.cal_nav_maxdrawdown(list(pct_sp_week))
                [nav_hs_week, maxdown_hs_week] = utils.cal_nav_maxdrawdown(list(pct_hs_week))
                [nav_gold_week, maxdown_gold_week] = utils.cal_nav_maxdrawdown(list(pct_gold_week))
                del nav_sh_week, nav_sp_week, nav_hs_week, nav_gold_week
                # 去掉第一个无涨跌幅的数据，防止影响分布
                maxdown_sh_week = np.array(maxdown_sh_week[1:])
                maxdown_sp_week = np.array(maxdown_sp_week[1:])
                maxdown_hs_week = np.array(maxdown_hs_week[1:])
                maxdown_gold_week = np.array(maxdown_gold_week[1:])

                # 当前所有资产的权重
                cur_weights = weights_origin.loc[inter_date]
                cur_sh_w = cur_weights['sh000300']
                cur_sp_w = cur_weights['SP500.SPI']
                cur_hs_w = cur_weights['HSCI.HI']
                cur_gold_w = cur_weights['GLNC']
                # 其它资产日期与沪深300对齐
                date_sp = utils.get_move_day(dates_sp, inter_date, 1)
                date_hs = utils.get_move_day(dates_hs, inter_date, 1)
                date_gold = utils.get_move_day(dates_gold, inter_date, 1)

                # 对各类资产做风控
                for cur_ass in self.assets:
                    if cur_ass == 'sh000300':
                        # 日回撤
                        # maxdown_now = maxdown_sh[-1]
                        # pre_maxdown_days = min(maxdown_sh[-252:-1])
                        # 当前时间点往前推一周的回撤
                        maxdown_now = maxdown_sh_week[-1]
                        # 当前时间点往前推一年（不包括当前周）的回撤
                        pre_maxdowns = maxdown_sh_week[:-1]
                        # 下面两行是做log，因为最大回撤都是<=0的，所以加abs(min(pre_maxdowns)) + 1.0，保证满足log要求
                        pre_maxdowns = np.log(pre_maxdowns + abs(min(pre_maxdowns)) + 1.0)
                        maxdown_now = np.log(maxdown_sh_week[-1] + abs(min(maxdown_sh_week)) + 1.0)
                        # 求最大回撤平均值
                        pre_md_mean = pre_maxdowns.mean()
                        # 求最大回撤标准差
                        pre_md_std = pre_maxdowns.std(ddof=1)
                        # 求置信空间
                        conf_int_95 = stats.norm.interval(self.sell_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(self.buy_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        # 求置信空间左侧上限
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)

                        # 下面的if语句是选择时间点画回撤分布和正太模拟分布图，不想画就把时间设置大于风控结束时间
                        exp_date = datetime.datetime(2020, 1, 17)
                        if inter_date == exp_date:
                            (mu, sigma) = stats.norm.fit(pre_maxdowns)
                            # s = np.random.normal(pre_md_mean, pre_md_std, 1000)
                            count, bins, ignored = plt.hist(pre_maxdowns, len(pre_maxdowns), normed=False)
                            # print type(bins)
                            # plt.plot(pre_maxdowns)
                            y = mtlab.normpdf(bins, mu, sigma)
                            y1 = mtlab.normpdf(bins, pre_md_mean, pre_md_std)
                            print pre_md_mean, pre_md_std
                            print mu, sigma
                            plt.plot(bins, y, 'r--', label='Norm. Dis.', linewidth=2)
                            # plt.plot(bins, y1, 'g--', label="Week Dis. Norm.", linewidth=2)
                            plt.legend(loc='upper left')
                            plt.xlabel("maxdown")
                            plt.ylabel("numbers")
                            plt.title('Hist of maxdown: mu='+str("%.3f" % mu)+',sigma='+str("%.3f" % sigma) \
                                      + '.97='+str("%.3f" % base_line_95) +',75='+str("%.3f" % base_line_75))
                            # plt.title('Hist of maxdown: mu1='+str("%.3f" % mu)+',sig1='+str("%.3f" % sigma) \
                            #           + ' mu2='+str("%.3f" % pre_md_mean) + ',sig2='+str("%.3f" % pre_md_std))
                            # plt.plot(bins, 1/(pre_md_std * np.sqrt(2 * np.pi)) * \
                            #          np.exp(-(bins - pre_md_mean)**2 / (2 * pre_md_std**2)), linewidth=2, color='r')
                            plt.show()
                            print pre_md_mean, pre_md_std, base_line_95, base_line_75
                            os._exit(0)

                        # 0.97风控开始
                        if maxdown_now < base_line_95 and not sh_risk:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            # 统计0.97风控六位数
                            risk_times += 1
                            # 0.97风控开始标志
                            sh_risk = True
                        # 0.97风控开始后操作
                        if sh_risk:
                            # 空仓
                            # sh_weight.append(0.0)
                            # sh_ttypes = np.append(sh_ttypes, 0.0)
                            # 0.97风控天数加1
                            sh_95.append(1)
                            sh_risk_days += 1
                            # 无条件空仓5天后
                            if sh_risk_days > 5:
                                #sh_risk == False
                                signal = self.tc_000300.loc[inter_date]['trade_types']
                                # 择时信号买入而且择时信号在此风控周期内未起作用
                                if signal == 1 and not sh_signal_effect:
                                    print inter_date, cur_ass + " signal effects"
                                    # sh_risk = False
                                    # sh_risk_days = 0
                                    sh_signal_effect = True
                                # 0.75风控点起作用
                                if maxdown_now >= base_line_75:
                                    print inter_date, cur_ass + " 75 effects"
                                    sh_risk = False
                                    sh_risk_days = 0
                                # 风控周期结束
                                if not sh_risk:
                                    sh_weight.append(cur_sh_w)
                                    sh_ttypes = np.append(sh_ttypes, 1.0)
                                    # 跳出风控周期后不再看择时信号
                                    sh_signal_effect = False
                                # 风控周期内看择时信号
                                elif sh_signal_effect:
                                    # 未达到1.25VaR，达到的话空仓
                                    if maxdown_now <= base_line_95 * 1.25:
                                        sh_weight.append(0.0)
                                        sh_ttypes = np.append(sh_ttypes, 0.0)
                                        # sh_signal_effect = False
                                    else:
                                        sh_weight.append(cur_sh_w)
                                        sh_ttypes = np.append(sh_ttypes, 1.0)
                                else:
                                    sh_weight.append(cur_sh_w)
                                    sh_ttypes = np.append(sh_ttypes, 1.0)
                                ## 择时信号起作用
                                #if sh_signal_effect:
                                #    # 未达到1.25VaR，达到的话空仓
                                #    if maxdown_now <= base_line_95 * 1.25:
                                #        sh_weight.append(0.0)
                                #        sh_ttypes = np.append(sh_ttypes, 0.0)
                                #    else:
                                #        sh_weight.append(cur_sh_w)
                                #        sh_ttypes = np.append(sh_ttypes, 1.0)
                                ## 风控周期未结束
                                #elif not sh_risk:
                                #    sh_weight.append(cur_sh_w)
                                #    sh_ttypes = np.append(sh_ttypes, 1.0)
                                #else:
                                #    sh_weight.append(0.0)
                                #    sh_ttypes = np.append(sh_ttypes, 0.0)
                            else:
                                sh_weight.append(0.0)
                                sh_ttypes = np.append(sh_ttypes, 0.0)
                        else:
                            sh_95.append(0)
                            sh_weight.append(cur_sh_w)
                            sh_ttypes = np.append(sh_ttypes, 1.0)
                        #if maxdown_now < pre_maxdown * 0.95 and sh_risk == False:
                        #    #sh_weight.append(0.0)
                        #    sh_risk = True
                    if cur_ass == 'SP500.SPI':
                        #maxdown_now = maxdown_sp[-1]
                        #pre_maxdown = min(maxdown_sp[-252:-1])
                        maxdown_now = maxdown_sp_week[-1]
                        pre_maxdowns = maxdown_sp_week[:-1]
                        pre_md_mean = pre_maxdowns.mean()
                        pre_md_std = pre_maxdowns.std()
                        conf_int_95 = stats.norm.interval(self.sell_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(self.buy_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        if maxdown_now < base_line_95 and not sp_risk:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            sp_risk = True
                        if sp_risk:
                            # sp_weight.append(0.0)
                            # sp_ttypes = np.append(sp_ttypes, 0.0)
                            sp_95.append(1)
                            sp_risk_days += 1
                            if sp_risk_days > 5:
                                #sh_risk == False
                                signal = self.tc_sp.loc[date_sp]['trade_types']
                                if signal == 1 and not sp_signal_effect:
                                    print inter_date, cur_ass + " signal effects"
                                    # sp_risk = False
                                    # sp_risk_days = 0
                                    sp_signal_effect = True

                                if maxdown_now > base_line_75:
                                    print inter_date, cur_ass + " 75 effects"
                                    sp_risk = False
                                    sp_risk_days = 0
                                if not sp_risk:
                                    sp_weight.append(cur_sp_w)
                                    sp_ttypes = np.append(sp_ttypes, 1.0)
                                    sp_signal_effect = False
                                elif sp_signal_effect:
                                    if maxdown_now <= base_line_95 * 1.25:
                                        sp_weight.append(0.0)
                                        sp_ttypes = np.append(sp_ttypes, 0.0)
                                    else:
                                        sp_weight.append(cur_sp_w)
                                        sp_ttypes = np.append(sp_ttypes, 1.0)
                                else:
                                    sp_weight.append(0.0)
                                    sp_ttypes = np.append(sp_ttypes, 0.0)
                            else:
                                sp_weight.append(0.0)
                                sp_ttypes = np.append(sp_ttypes, 0.0)
                        else:
                            sp_95.append(1)
                            sp_weight.append(cur_sp_w)
                            sp_ttypes = np.append(sp_ttypes, 1.0)
                        #if maxdown_now < pre_maxdown * 0.95 and sp_risk == False:
                        #    #sh_weight.append(0.0)
                        #    sp_risk = True
                    if cur_ass == 'HSCI.HI':
                        #maxdown_now = maxdown_hs[-1]
                        #pre_maxdown = min(maxdown_hs[-252:-1])
                        maxdown_now = maxdown_hs_week[-1]
                        pre_maxdowns = maxdown_hs_week[:-1]
                        pre_md_mean = pre_maxdowns.mean()
                        pre_md_std = pre_maxdowns.std()
                        conf_int_95 = stats.norm.interval(self.sell_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(self.buy_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        if maxdown_now < base_line_95 and hs_risk == False:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            hs_risk = True
                        if hs_risk:
                            # hs_weight.append(0.0)
                            # hs_ttypes = np.append(hs_ttypes, 0.0)
                            hs_95.append(1)
                            hs_risk_days += 1
                            if hs_risk_days > 5:
                                #sh_risk == False
                                signal = self.tc_hs.loc[date_hs]['trade_types']
                                if signal == 1 and not hs_signal_effect:
                                    print inter_date, cur_ass + " signal effects"
                                    # hs_risk = False
                                    # hs_risk_days = 0
                                    hs_signal_effect = True
                                if maxdown_now > base_line_75:
                                    print inter_date, cur_ass + " 75 effects"
                                    hs_risk = False
                                    hs_risk_days = 0

                                if not hs_risk:
                                    hs_weight.append(cur_hs_w)
                                    hs_ttypes = np.append(hs_ttypes, 1.0)
                                    hs_signal_effect = False
                                elif hs_signal_effect:
                                    if maxdown_now <= base_line_95 * 1.25:
                                        hs_weight.append(0.0)
                                        hs_ttypes = np.append(hs_ttypes, 0.0)
                                    else:
                                        hs_weight.append(cur_hs_w)
                                        hs_ttypes = np.append(hs_ttypes, 1.0)
                                else:
                                    hs_weight.append(0.0)
                                    hs_ttypes = np.append(hs_ttypes, 0.0)

                            else:
                                hs_weight.append(0.0)
                                hs_ttypes = np.append(hs_ttypes, 0.0)
                        else:
                            hs_95.append(0)
                            hs_weight.append(cur_hs_w)
                            hs_ttypes = np.append(hs_ttypes, 1.0)
                        #if maxdown_now < pre_maxdown * 0.95 and hs_risk == False:
                        #    #sh_weight.append(0.0)
                        #    hs_risk = True
                    if cur_ass == 'GLNC':
                        #maxdown_now = maxdown_gold[-1]
                        #pre_maxdown = min(maxdown_gold[-252:-1])
                        maxdown_now = maxdown_gold_week[-1]
                        pre_maxdowns = maxdown_gold_week[:-1]
                        pre_md_mean = pre_maxdowns.mean()
                        pre_md_std = pre_maxdowns.std()
                        conf_int_95 = stats.norm.interval(self.sell_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(self.buy_base_ratio, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        if maxdown_now < base_line_95 and not gold_risk:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            gold_risk = True
                        if gold_risk:
                            # gold_weight.append(0.0)
                            # gold_ttypes = np.append(gold_ttypes, 0.0)
                            gold_95.append(1)
                            gold_risk_days += 1
                            if gold_risk_days > 5:
                                #sh_risk == False
                                signal = self.tc_gold.loc[date_gold]['trade_types']
                                if signal == 1 and not gold_signal_effect:
                                    print inter_date, cur_ass + " signal effects"
                                    # gold_risk = False
                                    # gold_risk_days = 0
                                    gold_signal_effect = True
                                if maxdown_now > base_line_75:
                                    print inter_date, cur_ass + " 75 effects"
                                    gold_risk = False
                                    gold_risk_days = 0

                                if not gold_risk:
                                    gold_weight.append(cur_gold_w)
                                    gold_ttypes = np.append(gold_ttypes, 1.0)
                                    gold_signal_effect = False
                                elif gold_signal_effect:
                                    if maxdown_now <= base_line_95 * 1.25:
                                        gold_weight.append(0.0)
                                        gold_ttypes = np.append(gold_ttypes, 0.0)
                                    else:
                                        gold_weight.append(cur_gold_w)
                                        gold_ttypes = np.append(gold_ttypes, 1.0)
                                else:
                                    gold_weight.append(0.0)
                                    gold_ttypes = np.append(gold_ttypes, 0.0)
                            else:
                                gold_weight.append(0.0)
                                gold_ttypes = np.append(gold_ttypes, 0.0)
                        else:
                            gold_95.append(0)
                            gold_weight.append(cur_gold_w)
                            gold_ttypes = np.append(gold_ttypes, 1.0)
                        #if maxdown_now < pre_maxdown * 0.95 and gold_risk == False:
                        #    #sh_weight.append(0.0)
                        #    gold_risk = True
                if sh_risk or sp_risk or hs_risk or gold_risk:
                    total_95.append(1)
                else:
                    total_95.append(0)

            inter_date += datetime.timedelta(days=1)
        print "risk times:", risk_times
        sh_weight = np.array(sh_weight)
        sp_weight = np.array(sp_weight)
        hs_weight = np.array(hs_weight)
        gold_weight = np.array(gold_weight)

        sh_pct_m = sh_pct * (sh_ttypes > 0)
        sp_pct_m = sp_pct * (sp_ttypes > 0)
        hs_pct_m = hs_pct * (hs_ttypes > 0)
        gold_pct_m = gold_pct * (gold_ttypes > 0)

        sh_pct_c = sh_pct * sh_weight
        sp_pct_c = sh_pct * sp_weight
        hs_pct_c = sh_pct * hs_weight
        gold_pct_c = sh_pct * gold_weight

        sh_pct_o= sh_pct * sh_weight_ori
        sp_pct_o = sh_pct * sp_weight_ori
        hs_pct_o = sh_pct * hs_weight_ori
        gold_pct_o = sh_pct * gold_weight_ori

        total = sh_pct_c + sp_pct_c + hs_pct_c + gold_pct_c
        total_ori = sh_pct_o + sp_pct_o + hs_pct_o + gold_pct_o
        one_ratios = total
        mean_ratio = np.mean(one_ratios)
        anal_ratio = mean_ratio * 252.0
        return_std = np.std(one_ratios) * np.sqrt(252.0)
        sharpe_one = anal_ratio / return_std
        print "total anal_ratio:", anal_ratio
        print "total sharpe:", sharpe_one
        print "total variance:", return_std
        one_ratios = total_ori
        mean_ratio = np.mean(one_ratios)
        anal_ratio = mean_ratio * 252.0
        return_std = np.std(one_ratios) * np.sqrt(252.0)
        sharpe_one = anal_ratio / return_std
        print "total anal_ratio:", anal_ratio
        print "total_ori sharpe:", sharpe_one
        print "total_ori variance:", return_std

        [sh_nav_ori, sh_maxdown_ori] = utils.cal_nav_maxdrawdown(sh_pct)
        [sp_nav_ori, sp_maxdown_ori] = utils.cal_nav_maxdrawdown(sp_pct)
        [hs_nav_ori, hs_maxdown_ori] = utils.cal_nav_maxdrawdown(hs_pct)
        [gold_nav_ori, gold_maxdown_ori] = utils.cal_nav_maxdrawdown(gold_pct)

        [sh_nav, sh_maxdown] = utils.cal_nav_maxdrawdown(sh_pct_m)
        [sp_nav, sp_maxdown] = utils.cal_nav_maxdrawdown(sp_pct_m)
        [hs_nav, hs_maxdown] = utils.cal_nav_maxdrawdown(hs_pct_m)
        [gold_nav, gold_maxdown] = utils.cal_nav_maxdrawdown(gold_pct_m)

        [total_nav, total_maxdown] = utils.cal_nav_maxdrawdown(total)
        [total_nav_ori, total_maxdown_ori] = utils.cal_nav_maxdrawdown(total_ori)
        print "total maxdown:", min(total_maxdown)
        print "total_ori maxdown:", min(total_maxdown_ori)
        union_data = {}
        union_data['sh_nav_origin'] = sh_nav_ori
        union_data['sp_nav_origin'] = sp_nav_ori
        union_data['hs_nav_origin'] = hs_nav_ori
        union_data['gold_nav_origin'] = gold_nav_ori
        union_data['sh_maxdown_origin'] = sh_maxdown_ori
        union_data['sp_maxdown_origin'] = sp_maxdown_ori
        union_data['hs_maxdown_origin'] = hs_maxdown_ori
        union_data['gold_maxdown_origin'] = gold_maxdown_ori

        union_data['sh_nav'] = sh_nav
        union_data['sp_nav'] = sp_nav
        union_data['hs_nav'] = hs_nav
        union_data['gold_nav'] = gold_nav
        union_data['sh_maxdown'] = sh_maxdown
        union_data['sp_maxdown'] = sp_maxdown
        union_data['hs_maxdown'] = hs_maxdown
        union_data['gold_maxdown'] = gold_maxdown

        union_data['total_nav_origin'] = total_nav_ori
        union_data['total_maxdown_origin'] = total_maxdown_ori

        union_data['total_nav'] = total_nav
        union_data['total_maxdown'] = total_maxdown
        union_data['total_95'] = total_95
        union_data['sh_95'] = sh_95
        union_data = pd.DataFrame(union_data, index=dates)
        union_data.to_csv("risk_result.csv")

if __name__ == '__main__':
    tmpclass = RiskManagement()
    tmpclass.risk_control()
