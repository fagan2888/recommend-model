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
import jieba
import jieba.analyse
import os
import codecs
from .db import godeye_wechat_messages as wechat_mess
from .db import godeye_wechat_keywords as wechat_kw

def hprint(con):
    print(con)
    os._exit(0)
class WechatKeywords(object):
    def __init__(self, user_words, stop_words=False):
        # 开始有交易的时间
        self.is_user_words = user_words
        self.is_stop_words = stop_words
        self.start_date = wechat_mess.get_min_date()
        self.end_date = wechat_mess.get_max_date()

    def handle_by_week(self, s_date=None, e_date=None):
        """
        按周处理数据
        """
        if s_date is None:
            s_date = self.start_date
        if e_date is None:
            e_date = self.end_date
        dayscount = datetime.timedelta(days=self.start_date.isoweekday())
        dayfrom = self.start_date - dayscount + datetime.timedelta(days=1)
        timeDelta = datetime.timedelta(days=7)
        curday = dayfrom
        stopwords = self.stop_words()
        while curday <= e_date:
            print(curday)
            word_list = []
            word_dict = {}

            tmp_end = curday + timeDelta
            inter_data = wechat_mess.get_interval_data(curday, tmp_end)
            content_strs = ""
            for wd in inter_data:
                content_strs += wd[0]
            seg_rst = self.content_stem(content_strs)
            for wd in seg_rst:
                if self.is_stop_words:
                    if wd not in stopwords:
                        word_list.append(wd)
                else:
                    word_list.append(wd)

            for wd in word_list:
                if wd not in word_dict:
                    word_dict[wd] = 1
                else:
                    word_dict[wd] += 1
            tmp_dict = sorted(iter(list(word_dict.items())), key=lambda d:d[1], \
                reverse = False)
            self.update_db(tmp_dict, curday)

            curday = tmp_end

    def update_db(self, word_dict, cur_date):
        result_dict = {}
        wk_date = []
        wk_keywords = []
        wk_times = []
        wk_type = []
        for key, word in word_dict:
            wk_date.append(cur_date)
            wk_keywords.append(key)
            wk_times.append(word)
            wk_type.append(1)
        result_dict['wk_date'] = wk_date
        result_dict['wk_keywords'] = wk_keywords
        result_dict['wk_times'] = wk_times
        result_dict['wk_type'] = wk_type
        new_df = pd.DataFrame(result_dict).set_index([ \
        'wk_date', 'wk_keywords', 'wk_type'])
        new_df = new_df.ix[:, ['wk_times']]
        old_df = self.get_old_data(cur_date)
        wechat_kw.batch(new_df, old_df)

    def get_old_data(self, cur_date):
        old_data = wechat_kw.get_old_data(cur_date)
        if len(old_data) == 0:
            old_dict = {}
            old_dict['wk_date'] = []
            old_dict['wk_keywords'] = []
            old_dict['wk_times'] = []
            old_dict['wk_type'] = []
            old_df = pd.DataFrame(old_dict).set_index([ \
                    'wk_date', 'wk_keywords', 'wk_type'])
        else:
            old_df = pd.DataFrame(old_data)
            #old_df = old_df.iloc[:, :-1]
            old_df = old_df.set_index(['wk_date', 'wk_keywords', 'wk_type'])
        return old_df
    def content_stem(self, content):
        """
        对输入的内容分词
        :param content: 输入的文字,string
        :return: keywords, array()
        """
        user_words = self.ext_words()
        jieba.load_userdict(user_words)
        seg_list = jieba.cut(content, cut_all=True)
        return seg_list
    def stop_words(self):
        """
        加载过滤词
        """
        sfile =  "/home/yitao/recommend_model/asset_allocation_v1/stopword.dic";
        stopwords = [line.strip() for line in codecs.open( \
            sfile, 'r', 'utf8').readlines()]
        # for wd in stopwords:
        #     hprint(type(wd))
        return stopwords
    def ext_words(self):
        """
        加载自定义词
        """
        return "/home/yitao/recommend_model/asset_allocation_v1/ext.dic";
    def syn_words(self):
        """
        加载同义词
        """
        pass



if __name__ == "__main__":
    obj = WechatKeywords(True, True)
    #obj.get_old_data('2017-03-20')
    obj.handle_by_week()
