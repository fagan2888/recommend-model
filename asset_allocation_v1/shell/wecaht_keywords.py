# -*- coding: utf-8 -*-
"""
Created at Jun. 20, 2017
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import sys
sys.path.append('./shell')
import datetime
import calendar
import numpy as np
import pandas as pd
import os

from db import godeye_wechat_messages as wechat_mess

def hprint(con):
    print con
    os._exit(0)
class WechatKeywords(object):
    def __init__(self):
        # 开始有交易的时间
        self.start_date = wechat_mess.get_min_date()
        self.end_date = wechat_mess.get_max_date()
        hprint(self.end_date)
    def stop_words(self):
        """
        加载过滤词
        """
        pass
    def ext_words(self):
        """
        加载自定义词
        """
        pass
    def syn_words(self):
        """
        加载同义词
        """
        pass



if __name__ == "__main__":
    obj = WechatKeywords()
