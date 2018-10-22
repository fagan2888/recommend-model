#coding=utf8

import numpy as np
import pandas as pd
from ipdb import set_trace
from multiprocessing import Pool
from datetime import datetime

import sys
sys.path.append('shell/')
from db import base_ra_fund, base_fund_infos
from db import asset_fund


class MonetaryFundFilter:

    def __init__(self, yingmi_amount_limit=1e3):

        self.shholdercodes = [
            '10000001',  # 上海浦东发展银行股份有限公司
            '80001256',  # 上海银行股份有限公司
            '80001120',  # 中国农业银行股份有限公司
            '80001067',  # 中国工商银行股份有限公司
            '80001068',  # 中国建设银行股份有限公司
            '10000013',  # 中国民生银行股份有限公司
            '80001122',  # 中国银行股份有限公司
            '80001121',  # 交通银行股份有限公司
            '80001069',  # 兴业银行股份有限公司
            '80074105',  # 加拿大丰业银行
            '80086816',  # 加拿大皇家银行
            '80001097',  # 北京银行股份有限公司
            '80001483',  # 南京银行股份有限公司
            '80045209',  # 宁波银行股份有限公司
            '80001226',  # 恒生银行有限公司
            '10000020',  # 招商银行股份有限公司
            '80048262',  # 新加坡星展银行有限公司
            '80127394',  # 日本三井住友银行股份有限公司
            '80074107',  # 法国爱德蒙得洛希尔银行
            '80074107',  # 法国爱德蒙得洛希尔银行股份有限公司
            '80044863',  # 瑞士联合银行集团
            '80139440',  # 纽约银行梅隆资产管理国际有限公司
            '80049501',  # 苏格兰皇家银行有限公司
        ]

        self.blacklist_codes = ['80043419']  # 中融国际信托有限公司
        self.yingmi_amount_limit = yingmi_amount_limit

    def load_fund_info(self):

        fund_info = base_ra_fund.find_type_fund(3)

        return fund_info

    def load_fund_status(self):

        fund_status = base_fund_infos.load_status()
        blacklist_funds = asset_fund.load_fund_by_shholdercodes(self.blacklist_codes)
        valid_codes = np.setdiff1d(fund_status.fi_code.values, blacklist_funds.fsymbol.values)
        fund_status = fund_status[fund_status.fi_code.isin(valid_codes)]
        # fund_status = fund_status[fund_status.fi_yingmi_amount <= self.yingmi_amount_limit]

        return fund_status

    def load_scale(self, fund_codes):

        fund_share = asset_fund.load_share(fund_codes)
        now = datetime.now()
        today = datetime(now.year, now.month, now.day)
        fund_share.loc[today] = np.nan
        fund_share = fund_share.resample('d').last()
        fund_share = fund_share.fillna(method='pad')

        return fund_share

    def load_unit_nav(self, fund_codes):

        pool = Pool(32)
        results = pool.map(asset_fund.load_fund_unit_nav_series, fund_codes)
        pool.close()
        pool.join()
        fund_nav = dict(zip(fund_codes, results))
        fund_nav_df = pd.DataFrame(fund_nav)

        return fund_nav_df

    def handle(self):

        self.fund_info = self.load_fund_info()
        self.fund_status = self.load_fund_status()
        self.fund_scale = self.load_scale(self.fund_info.ra_code.values)
        self.fund_id_dict = dict(zip(self.fund_info.ra_code, self.fund_info.globalid))
        self.bank_funds = asset_fund.load_fund_by_shholdercodes(self.shholdercodes)


if __name__ == '__main__':

    mff = MonetaryFundFilter()
    mff.handle()

