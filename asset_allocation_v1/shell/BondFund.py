# -*- coding: UTF-8 -*-
import pandas as pd
import datetime
import numpy as np
import os

import BondFundPool
import utils

class BondFundFilter(object):
    def __init__(self):
        # 回测开始时间
        self.test_start = datetime.datetime(2015, 1, 1)
        # 回测结束时间
        self.test_end = datetime.datetime(2015, 12, 31)
        # 过滤的中间结果目录
        self.tmp_file = "../tmp/bond_filter_tmp.csv"
        # 基金类型
        self.types = [200301, 200302] #,200204 200306]
        # 债券基金过滤条件(0：所有条件, 1：基金类型，2：基金年限，3：基金份额，
        # 4：基金规模，5：基金经理，6：机构持有比例，7：基金中股票比例
        # 8：sharpe比率和波动率过滤)
        self.filter_type = [0]
        # 基金成立年限
        self.fund_age = 1

        # 基金规模
        self.fund_volume = 300000000.0
        self.fund_volume_new = 0.5
        self.fund_volume_pre_quarter = 0.5
        self.fund_volume_six_month_ago = 0.5
        self.fund_volume_found = 0.5

        # 基金份额
        self.share_ratio = 0.8
        # 基金经理
        self.manager_num = 5
        self.cur_manager_days = 360

        # 机构持有比例
        self.holding_ratio = 0.8
        self.holding_ratio_found = 0.5
        # sharpe计算时期
        self.days_one = 365
        self.days_two = 365 * 2
        self.days_three = 365 * 5
        # 股票市值占比
        self.stock_ratio = 15
        # sharpe/std
        self.sharpe = 1.0
        self.return_std = 0.1

    def filter_bond(self):
        self.filter_types()
        choose_date = self.test_start
        while choose_date <= self.test_end:
            self.filter_types(choose_date)
            for ftype in self.filter_type:
                if ftype == 0:
                    self.filter_types()
                    self.filter_found_years()
                    self.filter_share()
                    self.filter_volume()
                    self.filter_manager()
                    self.filter_ins_holding()
                    self.filter_stock_ratio()
                    self.filter_sharpe_std()
                    print "all filter apply"
                elif ftype == 1:
                    self.filter_types()
                    print "filter types"
                elif ftype == 2:
                    self.filter_found_years()
                    print "filter years"
                elif ftype == 3:
                    self.filter_share()
                    print "filter share"
                elif ftype == 4:
                    self.filter_volume()
                    print "filter volume"
                elif ftype == 5:
                    self.filter_manager()
                    print "filter manager"
                elif ftype == 6:
                    self.filter_ins_holding()
                    print "filter inst. "
                elif ftype == 7:
                    self.filter_stock_ratio()
                    print "filter stock holding ratio"
                elif ftype == 8:
                    self.filter_sharpe_std()
                    print "filter sharpe and std"
            choose_date = choose_date + datetime.timedelta(days=1)
        return None
    def filter_sharpe_std(self):
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], \
            parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_nav = pd.read_csv("../tmp/bondfunds_nav.csv", \
            index_col=['SECURITYID', 'NAVDATE'], parse_dates=['NAVDATE'])
        today = datetime.date.today()
        # now = datetime.datetime(today.year, today.month, today.day)
        one_year_delta = datetime.timedelta(days=self.days_one)
        two_year_delta = datetime.timedelta(days=self.days_two)
        three_year_delta = datetime.timedelta(days=self.days_three)
        pre_one_year = today - one_year_delta
        pre_two_year = today - two_year_delta
        pre_five_year = today - three_year_delta
        pre_one_year = datetime.datetime.strptime(str(pre_one_year),'%Y-%m-%d')
        pre_two_year = datetime.datetime.strptime(str(pre_two_year),'%Y-%m-%d')
        pre_five_year = datetime.datetime.strptime(str(pre_five_year),'%Y-%m-%d')
        sids = set(fund_base_info.index)
        fund_sharpe = {}
        one_sharpe = list()
        two_sharpe = list()
        five_sharpe = list()
        delete_list = set()
        for sid in sids:
            cur_fund_nav = fund_nav[fund_nav.index.get_level_values(0) == sid]
            cur_one_year_nav = cur_fund_nav[cur_fund_nav.index.get_level_values(1) >= pre_one_year]
            cur_two_year_nav = cur_fund_nav[cur_fund_nav.index.get_level_values(1) >= pre_two_year]
            cur_five_year_nav = cur_fund_nav[cur_fund_nav.index.get_level_values(1) >= pre_five_year]
            #cur_one_year_nav.fillna(0.0)
            #cur_two_year_nav.fillna(0.0)
            #cur_two_year_nav.fillna(0.0)
            #os._exit(0)
            cur_one_year_ratio = list(cur_one_year_nav.pct_change()['UNITNAV'])
            cur_one_year_ratio = cur_one_year_ratio[1:]

            cur_two_year_ratio = list(cur_two_year_nav.pct_change()['UNITNAV'])
            cur_two_year_ratio = cur_two_year_ratio[1:]

            cur_five_year_ratio = list(cur_five_year_nav.pct_change()['UNITNAV'])
            cur_five_year_ratio = cur_five_year_ratio[1:]

            one_ratios = cur_one_year_ratio
            mean_ratio = np.mean(one_ratios)
            anal_ratio = mean_ratio * 252.0
            return_std_one = np.std(one_ratios) * np.sqrt(252.0)
            sharpe_one = (anal_ratio - 0.03) / return_std_one

            one_ratios = cur_two_year_ratio
            mean_ratio = np.mean(one_ratios)
            anal_ratio = mean_ratio * 252.0
            return_std_two = np.std(one_ratios) * np.sqrt(252.0)
            sharpe_two = (anal_ratio - 0.03) / return_std_two
            #print one_ratios
            #print mean_ratio
            one_ratios = cur_five_year_ratio
            mean_ratio = np.mean(one_ratios)
            anal_ratio = mean_ratio * 252.0
            return_std_five = np.std(one_ratios) * np.sqrt(252.0)
            sharpe_five = (anal_ratio - 0.03) / return_std_five
            if sharpe_one < self.sharpe or sharpe_two < self.sharpe or sharpe_five < self.sharpe:
                delete_list.add(sid)
            if return_std_one > self.return_std or return_std_two > self.return_std \
                or return_std_five > self.return_std:
                delete_list.add(sid)
            print return_std_one, return_std_two, return_std_five
            one_sharpe.append(sharpe_one)
            two_sharpe.append(sharpe_two)
            five_sharpe.append(sharpe_five)

        fund_sharpe['one_year'] = one_sharpe
        fund_sharpe['two_year'] = two_sharpe
        fund_sharpe['five_year'] = five_sharpe
        #print one_sharpe
        #print two_sharpe
        #print five_sharpe
        print "filter sharpe ratio before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info.drop(list(delete_list))
        print "filter sharpe ratio after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')
        #fund_sharpe = pd.DataFrame(fund_sharpe, index=fund_base_info.index)
        #fund_sharpe.to_csv("../tmp/bondfunds_sharpe.csv")

    def filter_stock_ratio(self):
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], \
            parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_asset_port = pd.read_csv("../tmp/bondfunds_asset_port.csv", \
            index_col=['SECURITYID', 'REPORTDATE'], \
            parse_dates = ['REPORTDATE'])
        sids = set(fund_base_info.index)
        delete_list = set()
        for sid in sids:
            cur_asset_port = fund_asset_port[fund_asset_port.index.get_level_values(0) == sid]
            cur_asset_port = cur_asset_port.fillna(0.0)
            cur_asset_port = cur_asset_port['SKRATIO']
            ratio_one = cur_asset_port[-1]
            ratio_two = cur_asset_port[-2]
            ratio_three = cur_asset_port[-3]
            ratio_four = cur_asset_port[-4]
            ratio_found = cur_asset_port[0]
            if ratio_one > self.stock_ratio or ratio_two > self.stock_ratio \
                or ratio_three > self.stock_ratio \
                or ratio_four > self.stock_ratio:
                delete_list.add(sid)
        print "filter stock ratio before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info.drop(list(delete_list))
        print "filter stock ratio after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')
        return None
    def filter_ins_holding(self):
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], \
            parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_share_holding = pd.read_csv("../tmp/bondfunds_share_holding.csv", index_col=['SECURITYID', 'ENDDATE'], parse_dates = ['ENDDATE'])
        sids = set(fund_base_info.index)
        delete_list = set()
        for sid in sids:
            cur_share_holding = fund_share_holding[fund_share_holding.index.get_level_values(0) == sid]
            cur_share_holding = cur_share_holding.fillna(0.0)
            cur_share_holding = cur_share_holding['INVTOTRTO']
            holding_one = cur_share_holding[-1]
            holding_two = cur_share_holding[-2]
            holding_three = cur_share_holding[-3]
            holding_four = cur_share_holding[-4]
            holding_found = cur_share_holding[0]

            if holding_one < holding_two and holding_two < holding_three and \
                holding_three < holding_four and holding_four < holding_found:
                delete_list.add(sid)

            if holding_one <= holding_two * self.holding_ratio or \
                holding_two <= holding_three * self.holding_ratio or \
                holding_three <= holding_four * self.holding_ratio:
                delete_list.add(sid)
            if holding_one <= holding_found * self.holding_ratio_found:
                delete_list.add(sid)

        print "filter ins. holding before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info.drop(list(delete_list))
        print "filter ins. holding after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')
        return None
    def filter_manager(self):
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], \
            parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_manager = pd.read_csv("../tmp/bondfunds_manager.csv", \
            index_col=['COMPCODE','SECURITYID','BEGINDATE','PSCODE','POST'], \
            parse_dates = ['BEGINDATE','ENDDATE', 'ENTRYDATE'])
        sids = set(fund_base_info.index)
        delete_list = set()
        today = datetime.date.today()
        now = datetime.datetime(today.year, today.month, today.day)
        for sid in sids:
            fund_manager_filter = fund_manager[fund_manager.index.get_level_values(1) == sid]
            manager_num = len(set(fund_manager_filter.index.get_level_values(3)))
            fund_manager_filter_now = fund_manager_filter[fund_manager_filter['ISINCUMBENT'] == 1]
            manager_num_now = len(set(fund_manager_filter_now.index.get_level_values(3)))
            delta = now - fund_manager_filter_now.index.get_level_values('BEGINDATE')
            max_days = max(delta.days)
            if max_days < self.cur_manager_days:
                delete_list.add(sid)
            if manager_num >= self.manager_num and max_days <= self.cur_manager_days:
                delete_list.add(sid)
        print "filter manager before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info.drop(list(delete_list))
        print "filter manager after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')
        return None
    def filter_volume(self):
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_share = pd.read_csv("../tmp/bondfunds_share.csv", index_col=['ENDDATE'], parse_dates=['ENDDATE'])
        fund_nav = pd.read_csv("../tmp/bondfunds_nav.csv", index_col=['SECURITYID', 'NAVDATE'], parse_dates=['NAVDATE'])
        sids = set(fund_base_info.index)
        delete_list = set()
        nav_index_sids = set(fund_nav.index.get_level_values(0))

        for sid in sids:
            ssid = str(sid)
            # print ssid
            cur_nav_df = fund_nav[fund_nav.index.get_level_values(0) == sid]
            cur_nav_dates = cur_nav_df.index.get_level_values(1)
            fshare = fund_share[ssid]
            fshare = fshare.dropna(inplace=False)
            date_one = utils.get_move_day(cur_nav_dates, fshare.index[-1], 1)
            share_one = fshare[-1]
            date_two = utils.get_move_day(cur_nav_dates, fshare.index[-2], 1)
            share_two = fshare[-2]
            date_three = utils.get_move_day(cur_nav_dates, fshare.index[-3], 1)
            share_three = fshare[-3]
            date_four = utils.get_move_day(cur_nav_dates, fshare.index[-4], 1)
            share_four = fshare[-4]
            date_found = fshare.index[0]
            share_found = fshare[0]
            if sid in nav_index_sids:
                volume_one = share_one * fund_nav.loc[sid, date_one]['UNITNAV']
                volume_two = share_two * fund_nav.loc[sid, date_two]['UNITNAV']
                volume_three = share_three * fund_nav.loc[sid, date_three]['UNITNAV']
                volume_four = share_four * fund_nav.loc[sid, date_four]['UNITNAV']
                volume_found = share_found * 1.0 #fund_nav.loc[sid, date_found]['UNITNAV']
            else:
                print sid
                volume_one = share_one * 1.0
                volume_two = share_two * 1.0
                volume_three = share_three * 1.0
                volume_four = share_four * 1.0
                volume_found = share_found * 1.0

            if volume_one <= volume_two * self.fund_volume_new or \
                volume_two <= volume_three * self.fund_volume_pre_quarter or \
                volume_three <= volume_four * self.fund_volume_six_month_ago:
                delete_list.add(sid)

            if volume_one <= volume_found * self.fund_volume_found:
                delete_list.add(sid)

            if volume_three < volume_four and volume_four < volume_found:
                delete_list.add(sid)

            if volume_one < self.fund_volume or volume_two < self.fund_volume \
                or volume_three < self.fund_volume:
                delete_list.add(sid)

        print "filter volume before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info.drop(list(delete_list))
        print "filter volume after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')

    def filter_share(self):
        """
        按份额过滤
        """
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_share = pd.read_csv("../tmp/bondfunds_share.csv", index_col=['ENDDATE'], parse_dates=['ENDDATE'])
        sids = set(fund_base_info.index)
        # today = datetime.date.today()
        # now = datetime.datetime(today.year, today.month, today.day)
        delete_list = set()
        for sid in sids:
            ssid = str(sid)
            fshare = fund_share[ssid]
            fshare = fshare.dropna(inplace=False)
            share_one = fshare[-1]
            share_two = fshare[-2]
            share_three = fshare[-3]
            share_four = fshare[-4]
            share_found = fshare[0]
            if share_one < share_two * self.share_ratio or \
                share_two < share_three * self.share_ratio or \
                share_three < share_four * self.share_ratio:
                delete_list.add(sid)
            if share_one < share_two and share_two < share_three and share_three < share_four:
                delete_list.add(sid)
            if share_one < share_found * self.share_ratio:
                delete_list.add(sid)

        contains = sids - delete_list
        print "filter share before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info.drop(list(delete_list))
        print "filter share after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')

    def filter_types(self, choose_date):
        """
        按基金类型过滤
        """
        BondFundPool.load_bond_funds(self.types, choose_date.strftime("%Y%m%d"))
        fund_base_info = pd.read_csv("../tmp/bondfunds_base_info.csv", index_col=['SECURITYID'], parse_dates=['FOUNDDATE', 'ENDDATE'])
        fund_base_info.to_csv(self.tmp_file, encoding="utf8")

    def filter_found_years(self):
        """
        按基金成立时间长度过滤
        """
        fund_base_info = pd.read_csv(self.tmp_file, index_col=['SECURITYID'], parse_dates=['FOUNDDATE', 'ENDDATE'])
        indexs = fund_base_info.index
        one_year_ago = datetime.datetime.now() - datetime.timedelta(365*self.fund_age)
        print "filter year before: " + str(len(fund_base_info))
        fund_base_info = fund_base_info[fund_base_info['FOUNDDATE'] <= one_year_ago]
        print "filter year after: " + str(len(fund_base_info))
        fund_base_info.to_csv(self.tmp_file, index_col=['SECURITYID'], encoding='utf8')

if __name__ == "__main__":
    tmpclass = BondFundFilter()
    #tmpclass.filter_types()
    #tmpclass.filter_found_years()
    #tmpclass.filter_share()
    #tmpclass.filter_volume()
    #tmpclass.filter_manager()
    #tmpclass.filter_ins_holding()
    #tmpclass.filter_stock_ratio()
    #tmpclass.cal_sharpe()
    tmpclass.filter_bond()
