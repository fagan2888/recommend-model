# -*- coding: UTF-8 -*-
import pandas as pd
import datetime
import numpy as np
import utils
import os
from scipy import stats
class RiskManagement(object):

    def __init__(self):
        self.file_dir = "/home/data/yitao/recommend-model/recommend_model/asset_allocation_v1/tmp/"
        # 配置活跃幅
        self.port_pct = pd.read_csv(self.file_dir + "port_pct.csv", \
            index_col=['date'], parse_dates=['date'])
        # 配置比例
        self.port_weight = pd.read_csv(self.file_dir + "port_weight.csv", \
            index_col=['date'], parse_dates=['date'])
        # 沪深300择时信号
        self.tc_000300 = pd.read_csv(self.file_dir + "000300_gftd.csv", \
            index_col=['date'], parse_dates=['date'])
        self.tc_sp = pd.read_csv(self.file_dir + "sp_gftd.csv", \
            index_col=['date'], parse_dates=['date'])
        self.tc_hs = pd.read_csv(self.file_dir + "hs_gftd.csv", \
            index_col=['date'], parse_dates=['date'])
        self.tc_gold = pd.read_csv(self.file_dir + "gold_gftd.csv", \
            index_col=['date'], parse_dates=['date'])
        self.assets = ['sh000300', 'SP500.SPI', 'GLNC', 'HSCI.HI']
    def risk_control(self):
        dates = self.port_weight.index
        dates_sp = self.tc_sp.index
        dates_hs = self.tc_hs.index
        dates_gold = self.tc_gold.index

        sh_weight = []
        sp_weight = []
        hs_weight = []
        gold_weight = []

        sh_pct = []
        sp_pct = []
        hs_pct = []
        gold_pct = []

        weights_origin = self.port_weight.copy()
        pct_origin = self.port_pct.copy()

        start_time = dates[0]
        next_year = start_time + datetime.timedelta(days=365*1.5)
        next_year = datetime.datetime.strptime(next_year.strftime("%Y-%m-%d"), "%Y-%m-%d")
        test_start = utils.get_move_day(dates, next_year, 1)
        test_end = dates[-1]
        inter_date = test_start

        sh_weight = list(weights_origin[start_time:test_start]['sh000300'].values)[:-1]
        sp_weight = list(weights_origin[start_time:test_start]['SP500.SPI'].values)[:-1]
        hs_weight = list(weights_origin[start_time:test_start]['HSCI.HI'].values)[:-1]
        gold_weight = list(weights_origin[start_time:test_start]['GLNC'].values)[:-1]

        sh_weight_ori = np.array(weights_origin[start_time:test_end]['sh000300'].values)
        sp_weight_ori = np.array(weights_origin[start_time:test_end]['SP500.SPI'].values)
        hs_weight_ori = np.array(weights_origin[start_time:test_end]['HSCI.HI'].values)
        gold_weight_ori = np.array(weights_origin[start_time:test_end]['GLNC'].values)

        sh_pct = np.array(pct_origin[start_time:test_end]['sh000300'].values)
        sp_pct = np.array(pct_origin[start_time:test_end]['SP500.SPI'].values)
        hs_pct = np.array(pct_origin[start_time:test_end]['HSCI.HI'].values)
        gold_pct = np.array(pct_origin[start_time:test_end]['GLNC'].values)
        #pct_range = pct_origin[pct_origin.index.get_level_values(0) >= start_time]
        #pct_range = pct_range[pct_range.index.get_level_values(0) <= test_end]
        [nav_sh, maxdown_sh] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['sh000300'].values))
        [nav_sp, maxdown_sp] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['SP500.SPI'].values))
        [nav_hs, maxdown_hs] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['HSCI.HI'].values))
        [nav_gold, maxdown_gold] = utils.cal_nav_maxdrawdown(list(pct_origin[start_time:test_end]['GLNC'].values))
        #sh_pct_week = sh_pct[::5]
        #sp_pct_week = sp_pct[::5]
        #hs_pct_week = hs_pct[::5]
        #gold_pct_week = gold_pct[::5]
        #print min(maxdown_sh), min(maxdown_sp), min(maxdown_hs), min(maxdown_gold)
        maxdown_sh_ori = pd.DataFrame({"maxdown":maxdown_sh}, index=dates)
        maxdown_sp_ori = pd.DataFrame({"maxdown":maxdown_sp}, index=dates)
        maxdown_hs_ori = pd.DataFrame({"maxdown":maxdown_hs}, index=dates)
        maxdown_gold_ori= pd.DataFrame({"maxdown":maxdown_gold}, index=dates)

        nav_sh_ori = pd.DataFrame({"maxdown":maxdown_sh}, index=dates)
        nav_sp_ori = pd.DataFrame({"maxdown":maxdown_sp}, index=dates)
        nav_hs_ori = pd.DataFrame({"maxdown":maxdown_hs}, index=dates)
        nav_gold_ori= pd.DataFrame({"maxdown":maxdown_gold}, index=dates)

        sh_ttypes = np.ones(len(sh_weight))
        sp_ttypes = np.ones(len(sp_weight))
        hs_ttypes = np.ones(len(hs_weight))
        gold_ttypes = np.ones(len(gold_weight))

        sh_risk = False
        sp_risk = False
        hs_risk = False
        gold_risk = False

        sh_risk_days = 0
        sp_risk_days = 0
        hs_risk_days = 0
        gold_risk_days = 0
        risk_times = 0
        while inter_date <= test_end:

            if inter_date in dates:
                maxdown_sh = maxdown_sh_ori[maxdown_sh_ori.index.get_level_values(0)<=inter_date]['maxdown']
                maxdown_sp = maxdown_sp_ori[maxdown_sp_ori.index.get_level_values(0)<=inter_date]['maxdown']
                maxdown_hs = maxdown_hs_ori[maxdown_hs_ori.index.get_level_values(0)<=inter_date]['maxdown']
                maxdown_gold = maxdown_gold_ori[maxdown_gold_ori.index.get_level_values(0)<=inter_date]['maxdown']

                sh_pct_tmp = np.array(pct_origin[start_time:inter_date]['sh000300'].values)
                sp_pct_tmp = np.array(pct_origin[start_time:inter_date]['SP500.SPI'].values)
                hs_pct_tmp = np.array(pct_origin[start_time:inter_date]['HSCI.HI'].values)
                gold_pct_tmp = np.array(pct_origin[start_time:inter_date]['GLNC'].values)

                sh_pct_tmp = np.append([0.0], sh_pct_tmp[-5*54:])
                sp_pct_tmp = np.append([0.0], sp_pct_tmp[-5*54:])
                hs_pct_tmp = np.append([0.0], hs_pct_tmp[-5*54:])
                gold_pct_tmp = np.append([0.0], gold_pct_tmp[-5*54:])

                [nav_sh, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(sh_pct_tmp))
                [nav_sp, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(sp_pct_tmp))
                [nav_hs, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(hs_pct_tmp))
                [nav_gold, maxdown_tmp] = utils.cal_nav_maxdrawdown(list(gold_pct_tmp))

                nav_sh = nav_sh[::5]
                nav_sp = nav_sp[::5]
                nav_hs = nav_hs[::5]
                nav_gold = nav_gold[::5]

                pct_sh_week = np.diff(nav_sh)
                pct_sp_week = np.diff(nav_sp)
                pct_hs_week = np.diff(nav_hs)
                pct_gold_week = np.diff(nav_gold)

                pct_sh_week = np.append([0.0], pct_sh_week)
                pct_sp_week = np.append([0.0], pct_sp_week)
                pct_hs_week = np.append([0.0], pct_hs_week)
                pct_gold_week = np.append([0.0], pct_gold_week)

                [nav_sh_week, maxdown_sh_week] = utils.cal_nav_maxdrawdown(list(pct_sh_week))
                [nav_sp_week, maxdown_sp_week] = utils.cal_nav_maxdrawdown(list(pct_sp_week))
                [nav_hs_week, maxdown_hs_week] = utils.cal_nav_maxdrawdown(list(pct_hs_week))
                [nav_gold_week, maxdown_gold_week] = utils.cal_nav_maxdrawdown(list(pct_gold_week))

                maxdown_sh_week = np.array(maxdown_sh_week[1:])
                maxdown_sp_week = np.array(maxdown_sp_week[1:])
                maxdown_hs_week = np.array(maxdown_hs_week[1:])
                maxdown_gold_week = np.array(maxdown_gold_week[1:])

                cur_weights = weights_origin.loc[inter_date]
                cur_sh_w = cur_weights['sh000300']
                cur_sp_w = cur_weights['SP500.SPI']
                cur_hs_w = cur_weights['HSCI.HI']
                cur_gold_w = cur_weights['GLNC']
                date_sp = utils.get_move_day(dates_sp, inter_date, 1)
                date_hs = utils.get_move_day(dates_hs, inter_date, 1)
                date_gold = utils.get_move_day(dates_gold, inter_date, 1)
                for cur_ass in self.assets:
                    if cur_ass == 'sh000300':
                        #maxdown_now = maxdown_sh[-1]
                        #pre_maxdown = min(maxdown_sh[-252:-1])
                        maxdown_now = maxdown_sh_week[-1]
                        pre_maxdowns = maxdown_sh_week[:-1]
                        pre_md_mean = pre_maxdowns.mean()
                        pre_md_std = pre_maxdowns.std()
                        conf_int_95 = stats.norm.interval(0.95, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(0.75, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        #print pre_md_mean, pre_md_std, conf_int
                        #os._exit(0)
                        if maxdown_now < base_line_95 and sh_risk == False:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            sh_risk = True
                        if sh_risk:
                            sh_weight.append(0.0)
                            sh_ttypes = np.append(sh_ttypes, 0.0)
                            sh_risk_days += 1
                            if sh_risk_days >= 5:
                                #sh_risk == False
                                signal = self.tc_000300.loc[inter_date]['trade_types']
                                if signal == 1:
                                    sh_risk = False
                                    sh_risk_days = 0
                                elif maxdown_now > base_line_75:
                                    sh_risk = False
                                    sh_risk_days = 0
                        else:
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
                        conf_int_95 = stats.norm.interval(0.95, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(0.75, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        if maxdown_now < base_line_95 and sp_risk == False:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            sp_risk = True
                        if sp_risk:
                            sp_weight.append(0.0)
                            sp_ttypes = np.append(sp_ttypes, 0.0)
                            sp_risk_days += 1
                            if sp_risk_days >= 5:
                                #sh_risk == False
                                signal = self.tc_sp.loc[date_sp]['trade_types']
                                if signal == 1:
                                    sp_risk = False
                                    sp_risk_days = 0
                                elif maxdown_now > base_line_75:
                                    sp_risk = False
                                    sp_risk_days = 0
                        else:
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
                        conf_int_95 = stats.norm.interval(0.95, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(0.75, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        if maxdown_now < base_line_95 and hs_risk == False:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            hs_risk = True
                        if hs_risk:
                            hs_weight.append(0.0)
                            hs_ttypes = np.append(hs_ttypes, 0.0)
                            hs_risk_days += 1
                            if hs_risk_days >= 5:
                                #sh_risk == False
                                signal = self.tc_hs.loc[date_hs]['trade_types']
                                if signal == 1:
                                    hs_risk = False
                                    hs_risk_days = 0
                                elif maxdown_now > base_line_75:
                                    hs_risk = False
                                    hs_risk_days = 0
                        else:
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
                        conf_int_95 = stats.norm.interval(0.95, loc=pre_md_mean, scale=pre_md_std)
                        conf_int_75 = stats.norm.interval(0.75, loc=pre_md_mean, scale=pre_md_std)
                        base_line_95 = min(conf_int_95)
                        base_line_75 = min(conf_int_75)
                        if maxdown_now < base_line_95 and gold_risk == False:
                            #sh_weight.append(0.0)
                            print inter_date, cur_ass
                            risk_times += 1
                            gold_risk = True
                        if gold_risk:
                            gold_weight.append(0.0)
                            gold_ttypes = np.append(gold_ttypes, 0.0)
                            gold_risk_days += 1
                            if gold_risk_days >= 5:
                                #sh_risk == False
                                signal = self.tc_gold.loc[date_gold]['trade_types']
                                if signal == 1:
                                    gold_risk = False
                                    gold_risk_days = 0
                                elif maxdown_now > base_line_75:
                                    gold_risk = False
                                    gold_risk_days = 0
                        else:
                            gold_weight.append(cur_gold_w)
                            gold_ttypes = np.append(gold_ttypes, 1.0)
                        #if maxdown_now < pre_maxdown * 0.95 and gold_risk == False:
                        #    #sh_weight.append(0.0)
                        #    gold_risk = True

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
        print "total sharpe:", sharpe_one
        print "total variance:", return_std
        one_ratios = total_ori
        mean_ratio = np.mean(one_ratios)
        anal_ratio = mean_ratio * 252.0
        return_std = np.std(one_ratios) * np.sqrt(252.0)
        sharpe_one = anal_ratio / return_std
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
        union_data = pd.DataFrame(union_data, index=dates)
        union_data.to_csv("../tmp/risk_result.csv")

if __name__ == '__main__':
    tmpclass = RiskManagement()
    tmpclass.risk_control()
