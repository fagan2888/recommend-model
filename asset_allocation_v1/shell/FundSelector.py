#coding=utf8


import sys
sys.path.append("windshell")
import main
import Data
import Const
import string
from numpy import *
import numpy as np
import pandas as pd
import Financial as fin
import FundIndicator as fi


def select_stock(funddf, fund_tags, indexdf):

    largecap_codes             = fund_tags['largecap']
    smallcap_codes             = fund_tags['smallcap']
    risefitness_codes          = fund_tags['risefitness']
    declinefitness_codes       = fund_tags['declinefitness']
    oscillationfitness_codes   = fund_tags['oscillationfitness']
    growthfitness_codes        = fund_tags['growthfitness']
    valuefitness_codes         = fund_tags['valuefitness']


    need_largecap           = True
    need_smallcap           = True
    need_risefitness        = True
    need_declinefitness     = True
    need_oscillationfitness = True
    need_growthfitness      = True
    need_valuefitness       = True

    #fund_sharpe  = fi.fund_sharp_annual(funddf)
    fund_sharpe  = fi.fund_jensen(funddf, indexdf)

    codes = []
    tag   = {}
    for i in range(0, len(fund_sharpe)):

        code = fund_sharpe[i][0]

        if code in set(largecap_codes) and need_largecap:
            codes.append(code)
            need_largecap = False
            tag['largecap'] = code
            #print code, i
            #continue
        if code in set(smallcap_codes) and need_smallcap:
            codes.append(code)
            need_smallcap = False
            tag['smallcap'] = code
            #print code, i
            #continue
        if code in set(risefitness_codes) and need_risefitness:
            codes.append(code)
            need_risefitness = False
            tag['rise'] = code
            #print code, i
            #continue
        if code in set(declinefitness_codes) and need_declinefitness:
            codes.append(code)
            need_declinefitness = False
            tag['decline'] = code
            #print code, i
            #continue
        if code in set(oscillationfitness_codes) and need_oscillationfitness:
            codes.append(code)
            need_oscillationfitness = False
            tag['oscillation'] = code
            #print code, i
            #continue
        if code in set(growthfitness_codes) and need_growthfitness:
            codes.append(code)
            need_growthfitness = False
            tag['growth'] = code
            #print code, i
            #continue
        if code in set(valuefitness_codes) and need_valuefitness:
            codes.append(code)
            need_valuefitness = False
            tag['value'] = code
            #print code, i
            #continue

    #print fund_sharpe
    #print codes

    return codes, tag


def select_bond(funddf, fund_tags, indexdf):

    ratebond_codes             = fund_tags['ratebond']
    credit_codes               = fund_tags['creditbond']
    convertible_codes          = fund_tags['convertiblebond']

    need_rate             = True
    need_credit           = True
    need_convertible      = True


    #fund_sharpe  = fi.fund_sharp_annual(funddf)
    fund_sharpe  = fi.fund_jensen(funddf, indexdf)
    codes = []
    tag   = {}
    for i in range(0, len(fund_sharpe)):

        code = fund_sharpe[i][0]

        if code in set(ratebond_codes) and need_rate:
            codes.append(code)
            need_largecap = False
            tag['ratebond'] = code
            #print code, i
            continue
        if code in set(credit_codes) and need_credit:
            codes.append(code)
            need_credit = False
            tag['creditbond'] = code
            #print code, i
            continue
        if code in set(convertible_codes) and need_convertible:
            codes.append(code)
            need_convertible = False
            tag['convertiblebond'] = code
            #print code, i
            continue

    #print fund_sharpe
    #print codes

    return codes, tag


def select_money(funddf):

    fund_sharpe  = fi.fund_sharp_annual(funddf)
    codes = []
    tag   = {}
    tag['money'] = fund_sharpe[0][0]
    codes.append(fund_sharpe[0][0])
    #tag['sharpe2'] = fund_sharpe[1][0]
    #codes.append(fund_sharpe[0][0])

    return codes, tag



