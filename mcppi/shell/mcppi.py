#!/usr/bin/python
#coding=utf8

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import datetime

#第0部分：说明
#1、只考虑自然天数，不考虑交易天数
#2、输入数据以周为索引
#3、固定资产投资周期3个月，浮动资产投资周期1个月，现金资产投资周期1周

#第1部分：输入
cap_init = 10000.#初始总金额
ret_free=.03#无风险收益率
cou_ite_fixed=90#固定收益资产调整周期（天）
cou_dayinyear=360#一年天数，自然天数下为360，交易天数下一般为260
retio_currency=.02#现金资产（占除固定资产之外的剩余资产的）比例底限        

p = pd.read_excel('c:/Users/spock/Desktop/mcppi_input_all.xlsx', index_col='date', parse_dates=True)
dates = p.index#日期序列，从参数文件中读出

#第2部分：处理
arylen=len(dates)
#用数组记录每一期的三类资产投资额度
ary_fixed = [0]*arylen
ary_floating = [0]*arylen
ary_currency= [0]*arylen

ite_cur=0#日期序号
num_fixed=0.#固定资产投资额
num_floating=0.#浮动资产投资额
num_currency=0.#现金资产投资额
cap_cur=cap_init#每一期的期初总金额
#for date in [datetime.datetime.strptime(d,'%Y-%m-%d') for d in dates]:
for d in dates:

    #每一期都要预先准备几个参数，从参数文件中读出（每一期都不同）
    sec_mul=p['sec_mul'][d]#本期风险乘数
    sec_bottom_floating=p['sec_bottom_floating'][d]#本期浮动资产安全垫
    ret_fixed=p['ret_fixed'][d]#固定资产收益率
    ret_floating=p['ret_floating'][d]#浮动资产收益率
    ret_currency=p['ret_currency'][d]#现金资产收益率
    
    #固定资产调整周期最长，所以先算固定资产
    if(ite_cur % 12 ==0):#每3个月为一个固定资产投资期，3个月=12个周
        #价值底线
        value_bottom=cap_cur/(1.+ret_free)**(cou_ite_fixed*(ite_cur/12+1)/cou_dayinyear)
        #固定资产安全垫
        sec_bottom=cap_cur-value_bottom
        #能够用于浮动投资和现金投资的金额
        num_nofixed=sec_bottom*(2 if ite_cur==0 else sec_mul)
        #固定资产投资额
        num_fixed=cap_cur-num_nofixed#固定资产投资额确定后，本期内不再变化
    
    #浮动资产调整周期短于固定资产，但长于现金资产，所以其次算浮动资产
    if(ite_cur % 4 ==0):#每1个月为一个浮动资产投资期，1个月=4个周
        
        #计算拟定的浮动资产投资额上限max_fixed
        max_fixed=0.
        if(ite_cur % 12 ==0):
            max_fixed=num_nofixed
        else:
            if cap_cur > sec_bottom_floating:#如果当前总资产高于安全垫，则可以把部分资产投资于浮动资产
                max_fixed = (cap_cur-sec_bottom_floating)*sec_mul    
                if max_fixed/cap_cur > .4:#浮动资产的投资额上限不能超过当期总资产的40%
                    max_fixed = cap_cur*.4
            else:#如果当前总资产少于安全垫，则不投资于浮动资产
                max_fixed=0.
    else:
        #计算拟定的浮动资产投资额上限max_fixed
        max_fixed=num_floating#在非调整期，浮动资产投资上限等于上一期浮动资产投资额
    #最终的本期浮动资产投资额
    num_floating=max_fixed if max_fixed<(cap_cur-num_fixed)*(1-retio_currency) else (cap_cur-num_fixed)*(1-retio_currency)
    
    #最后算现金资产
    num_currency=cap_cur-num_fixed-num_floating
    
    ary_fixed[ite_cur] = num_fixed
    ary_floating[ite_cur] = num_floating
    ary_currency[ite_cur] = num_currency
    print(str(d)+'：'+str(cap_cur)+'\t'+str(num_fixed)+'\t'+str(num_floating)+'\t'+str(num_currency))
    
    #计算本期期末总金额
    income_fixed=ret_fixed*num_fixed#本期固定资产收益
    income_floating=ret_floating*num_floating#本期固定资产收益
    income_currency=ret_currency*num_currency#本期固定资产收益
    cap_cur = cap_cur+(income_fixed+income_floating+income_currency)
    
    ite_cur = ite_cur+1
    
    
#第3部分：输出
#f = io.open('/home/spock/Downloads/2016021901.out.txt', "w")
#f.write(unicode(np.round(cash_p[i], 3))+'\t'+unicode(np.round(inv_p[i], 3))+'\t'+unicode(np.round(ins_p[i], 3))+"\t"+unicode(np.round(sr_list[i], 8))+"\n")
#f.close()


