#coding=utf8


import string
import numpy as np
import sys
sys.path.append('windshell')
import const
import Financial as fin
import fundindicator as fi


def evaluation(funddf, pvs):


	fundsharp     =    fi.fund_sharp(funddf)
	fundreturn    =    fi.fund_return(funddf)
	fundrisk      =    fi.fund_risk(funddf)

	#print fundsharp

	middlesharp   =    fundsharp[len(fundsharp) / 2][1]   
	middlereturn  =	   fundreturn[len(fundreturn) / 2][1]
	middlerisk    =    fundrisk[len(fundrisk) / 2][1]   	 


	prs           =    []
	for i in range(1, len(pvs)):
		prs.append(pvs[i] / pvs[i-1] - 1)		


	
	psharp        =    fi.portfolio_sharp(prs)
	preturn       =    fi.portfolio_return(prs)
	prisk         =    fi.portfolio_risk(prs)



	title_str = '%20s, %15s, %15s, %15s' % ('', 'sharp' ,'return', 'risk')
	p_str     = '%20s, %15s, %15s, %15s' % ('portfolio', str(psharp) ,str(preturn), str(prisk))
	fund_str  = '%20s, %15s, %15s, %15s' % ('market middle', str(middlesharp) ,str(middlereturn), str(middlerisk))


	print title_str
	print p_str
	print fund_str	



