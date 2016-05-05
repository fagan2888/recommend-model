#coding=utf8


import sys
sys.path.append("windshell")
import main
import data
import const
import string
from numpy import *
import numpy as np
import pandas as pd
import Financial as fin
import fundindicator as fi


def select_fund(funddf, fund_tags):

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

	fund_sharpe  = fi.fund_sharp_annual(funddf)
	codes = []
	for i in range(0, len(fund_sharpe)):

		code = fund_sharpe[i][0]

		if code in set(largecap_codes) and need_largecap:
			codes.append(code)
			need_largecap = False
			print code, i
			continue
		if code in set(smallcap_codes) and need_smallcap:
			codes.append(code)
			need_smallcap = False
			print code, i
			continue
		if code in set(risefitness_codes) and need_risefitness:
			codes.append(code)
			need_risefitness = False
			print code, i
			continue
		if code in set(declinefitness_codes) and need_declinefitness:
			codes.append(code)
			need_declinefitness = False
			print code, i
			continue
		if code in set(oscillationfitness_codes) and need_oscillationfitness:
			codes.append(code)
			need_oscillationfitness = False
			print code, i
			continue
		if code in set(growthfitness_codes) and need_growthfitness:
			codes.append(code)
			need_growthfitness = False
			print code, i
			continue
		if code in set(valuefitness_codes) and need_valuefitness:
			codes.append(code)
			need_valuefitness = False
			print code, i
			continue

	#print fund_sharpe
	#print codes

	return codes



