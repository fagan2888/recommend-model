#coding=utf8

import string
from numpy import *
import numpy as np
import pandas as pd



#大盘适应度
def largecaptag(funddf, indexdf):

	return 0


#小盘适应度
def smallcaptag():
	return 0


#上涨适应度
def risetag(funddf, indexdf):

	funddfr = funddf.pct_change()
        indexdfr = funddf.pct_change()

        risetag = {}

	rise = []	

	indexr = indexdfr.values

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1


		if n == 0:
			rise.append(0)
		elif n > 0:
			rise.append(1)
		else:
			rise.append(-1)																		

	for code in funddfr.volumns:
	
		fundr = funddfr[code].values				
		riser = []
		for i in range(4, len(rise)):
			tag = rise[i - 4]
			if tag == 1:
				riser.append(fundr[i])
	
		risetag[code] = riser			
												
	return risetag



#下跌适应度
def declinetag():

	funddfr = funddf.pct_change()
        indexdfr = funddf.pct_change()

        declinetag = {}

	decline = []	

	indexr = indexdfr.values

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			decline.append(0)
		elif n > 0:
			decline.append(1)
		else:
			decline.append(-1)																		

	for code in funddfr.volumns:
	
		fundr = funddfr[code].values				
		decliner = []
		for i in range(4, len(decline)):
			tag = decline[i - 4]
			if tag == -1:
				decliner.append(fundr[i])
	
		declinetag[code] = decliner	
												
	return declinetag


#震荡适应度
def oscillationtag():

	funddfr = funddf.pct_change()
        indexdfr = funddf.pct_change()

        oscillationtag = {}

	oscillation = []	

	indexr = indexdfr.values

	for i in range(4, len(indexr)):
		n = 0
		for j in range(0, 4):
			v = indexr[i - j]
			if v >= 0:
				n = n + 1
			else:
				n = n - 1

		if n == 0:
			oscillation.append(0)
		elif n > 0:
			oscillation.append(1)
		else:
			oscillation.append(-1)																		
	for code in funddfr.volumns:
	
		fundr = funddfr[code].values				
		oscillationr = []
		for i in range(4, len(rise)):
			tag = oscillation[i - 4]
			if tag == 0:
				oscillationr.append(fundr[i])
	
		oscillationtag[code] = oscillationr			
												
	return risetag


#成长适应度
def growth():
	return 0


#价值适应度
def value():
	return 0 



