# -*- coding: UTF-8 -*-

Class BondFundFilter(object):
    def __init__(self, bf_df):
        # 基金成立年限
        self.fund_age = 1

        # 基金规模
        self.fund_found = 10
        self.fund_volume_new = 3
        self.fund_volume_pre_quarter = 3
        self.fund_volume_six_month_ago = 3
        self.fund_volume_triple_quarter_ago = 3

        # 规模比例过滤
        self.fund_ratio_pre_quarter = 50
        self.fund_ratio_six_month_ago = 50
        self.fund_ratio_triple_quarter_ago = 50

        
