#coding=utf8


from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
import numpy as np


class Markowitz:

	#
	#注释
	#return_rate : 资产收益率矩阵
	def __init__(self, return_rate):
		self.r     =     return_rate

	#
	#获得有效前沿
	def efficient_frontier(self):

		n_asset    =     len(self.r)
		asset_mean = []
		for record in self.r:
			asset_mean.append(np.mean(record))
		cov        =     np.cov(self.r)
		
		S	   =     matrix(cov)
		pbar       =     matrix(asset_mean)	
		
		G          =     matrix(0.0, (n_asset, n_asset))	
		G[::n_asset + 1]  =  -1.0	
		h                 =  matrix(0.0, (n_asset, 1))
		A                 =  matrix(1.0, (1, n_asset))
		b                 =  matrix(1.0)
				
		N = 1000
		mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
		portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
		returns = [ dot(pbar,x) for x in portfolios ]
		risks = [ sqrt(dot(x, S*x)) for x in portfolios ]	
		
		return risks, returns, portfolios



if __name__ == '__main__':


	rs = [[0.25,0.3,0.4, 0.3, 0.2, -0.1, -0.2], [0.1, 0.2, 0.3, -0.01, -0.2, 0.01, 0.02]]	
	bond = [[0, 1.0],[0, 1.0]]
	m  = Markowitz(rs)
	risks, returns, portfolios = m.efficient_frontier()

	#	print risks
	#	print returns
	#	print portfolios[999]
	#for i in range(0, len(risks)):
	#	print risks[i], ',', returns[i]	
