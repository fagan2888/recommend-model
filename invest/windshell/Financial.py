#coding=utf8


import string
import math
from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
from sklearn import datasets, linear_model	
import numpy as np
from scipy.stats import norm
from cvxopt import matrix, solvers
from numpy import isnan
from scipy import linalg



#计算有效前沿
def efficient_frontier(return_rate, bound):

	solvers.options['show_progress'] = False

	n_asset    =     len(return_rate)

	asset_mean = np.mean(return_rate, axis = 1)
	#print asset_mean

	cov        =     np.cov(return_rate)
	
	S	   =     matrix(cov)
	pbar       =     matrix(asset_mean)	

	if bound == None or len(bound) == 0:


		G          =     matrix(0.0, (n_asset, n_asset))
		G[::n_asset + 1]  =  -1.0
		h                 =  matrix(0.0, (n_asset, 1))
		A                 =  matrix(1.0, (1, n_asset))
		b                 =  matrix(1.0)


		N = 100
		mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
		portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
		returns = [ dot(pbar,x) for x in portfolios ]
		risks = [ sqrt(dot(x, S*x)) for x in portfolios ]


		return risks, returns, portfolios


	else:

		G          =     matrix(0.0 , (3 * n_asset,  n_asset))

		for i in range(0, n_asset):
			G[i, i] = -1
			G[n_asset + i, i ] = -1
			G[2 * n_asset + i, i ] = 1


		h                 =  matrix(0.0, (3 * n_asset, 1))


		for i in range(0, n_asset):
			h[n_asset + i, 0] = -1.0 * bound[0][i]
			h[2 * n_asset + i, 0] = bound[1][i]
	
		
		A                 =  matrix(1.0, (1, n_asset))
		b                 =  matrix(1.0)

				
		N = 100
		mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
		portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
		returns = [ dot(pbar,x) for x in portfolios ]
		risks = [ sqrt(dot(x, S*x)) for x in portfolios ]	

	
		return risks, returns, portfolios




#计算有效前沿
def efficient_frontier_index(return_rate):

	solvers.options['show_progress'] = False
        
	n_asset    =     len(return_rate)

        asset_mean = np.mean(return_rate, axis = 1)
        #print asset_mean

        cov        =     np.cov(return_rate)

        S          =     matrix(cov)
        pbar       =     matrix(asset_mean)

        G          =     matrix(0.0, (n_asset, n_asset))
        G[::n_asset + 1]  =  -1.0
        h                 =  matrix(0.0, (n_asset, 1))
        A                 =  matrix(1.0, (1, n_asset))
        b                 =  matrix(1.0)


        N = 100
        mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
        portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
        returns = [ dot(pbar,x) for x in portfolios ]
        risks = [ sqrt(dot(x, S*x)) for x in portfolios ]


        return risks, returns, portfolios


#计算有效前沿
def efficient_frontier_fund(return_rate):

	solvers.options['show_progress'] = False

	n_asset    =     len(return_rate)

	asset_mean = np.mean(return_rate, axis = 1)
	#print asset_mean

	cov        =     np.cov(return_rate)
	
	S	   =     matrix(cov)
	pbar       =     matrix(asset_mean)	

	G          =     matrix(0.0 , (2 * n_asset,  n_asset))

	for i in range(0, n_asset):
		G[i, i] = -1
		G[n_asset + i, i ] = 1


	h                 =  matrix(0.0, (2 * n_asset, 1))


	for i in range(0, n_asset):
		h[n_asset + i, 0] = 0.5
	
	
	A                 =  matrix(1.0, (1, n_asset))
	b                 =  matrix(1.0)

			
	N = 100
	mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
	portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
	returns = [ dot(pbar,x) for x in portfolios ]
	risks = [ sqrt(dot(x, S*x)) for x in portfolios ]	

	
	return risks, returns, portfolios


#计算下方差
def semivariance(portfolio):
	mean       =        np.mean(portfolio)
	var        =        0.0
	n          =        0	
	for d in portfolio:
		if d <= mean:
			n   = n + 1
			var = var + (d - mean) ** 2
	var        =    var / n

	return     math.sqrt(var)	
			
		
#jensen测度
'''
def jensen(portfolio, market, rf):

	pr         =    np.mean(portfolio)
	mr         =    np.mean(market)
	beta       =    np.cov(portfolio, market)[0][1] / np.cov(market)
	return pr - (rf + beta * ( mr - rf) )	
'''


def jensen(portfolio, market, rf):
	
	p = []
	m = []		
	for i in range(0, len(portfolio)):
		p.append(portfolio[i] - rf)
		m.append([market[i] - rf])

	#print p
	#print m	
	clf       = linear_model.LinearRegression()
	clf.fit(m, p)
	alpha = clf.intercept_
	beta  = clf.coef_[0]
	return alpha


			
#sharp
def sharp(portfolio, rf):
	r         =    np.mean(portfolio)
	sigma     =    np.std(portfolio)
	return    (r - rf) / sigma



#索提诺比率
def sortino(portfolio, rf):
	pr        =    np.mean(portfolio)
	return (pr - rf ) / semivariance(portfolio)



#treynor-mazuy模型
def tm(portfolio, market, rf):
	xparams   = []	
	yparams   = []
	for i in range(0, len(portfolio)):
		yparams.append(portfolio[i] - rf)
		xparams.append([market[i] - rf, (market[i] - rf) ** 2])					
		
	clf       = linear_model.LinearRegression()
	clf.fit(xparams, yparams)
	return clf.intercept_,  clf.coef_[0], clf.coef_[1]


#henrikson-merton
def hm(portfolio, market, rf):
	xparams   = []	
	yparams   = []
	for i in range(0, len(portfolio)):
		yparams.append(portfolio[i] - rf)
		if rf - market[i] > 0:
			xparams.append([market[i] - rf, market[i] - rf])					
		else:
			xparams.append([market[i] - rf, 0])	
					
	clf       = linear_model.LinearRegression()
	clf.fit(xparams, yparams)
	return clf.intercept_,  clf.coef_[0], clf.coef_[1]


#value at risk
def var(portfolio):
	parray = np.array(portfolio)	
	valueAtRisk = norm.ppf(0.05, parray.mean(), parray.std())	
	return valueAtRisk



#positive peroid weight
def ppw(portfolio, benchmark):	

	
	#print 'p', portfolio
	#print 'm', benchmark

	
	length = len(benchmark)
	A = []
	b = []

	for i in range(0, length):
		item = []
		for j in range(0, length + 3):
			item.append(0)
		A.append(item)
	
	for i in range(0,  length + 3):
		b.append(0)	

	for i in range(0, length):
		A[i][i] = -1
		b[i]    = -1e-13

	i = length
	for j in range(0, length):
		A[j][i]	 = 1
	b[i] = 1	


	i = length + 1				
	for j in range(0, length):
		A[j][i] = -1
	b[i] = -1	


	i = length + 2
	for j in range(0, length):
		A[j][i] = -benchmark[j]

	b[i] = 0		 
			
	c = []
	for j in range(0, length):
		c.append(benchmark[j])	


	A = matrix(A)
	b = matrix(b)
	c = matrix(c)

	#print A
	#print b
	#print c

	sol = solvers.lp(c, A, b)
	ppw = 0
	for i in range(0, len(sol['x'])):
		ppw = ppw + portfolio[i] * sol['x'][i]	

	return ppw





def grs(portfolio):
	return 0	




def black_litterman(delta, weq, sigma, tau, P, Q, Omega):

	# Reverse optimize and back out the equilibrium returns
	# This is formula (12) page 6.
	#print weq


	pi = weq.dot(sigma * delta)


	# We use tau * sigma many places so just compute it once
	ts = tau * sigma


	# Compute posterior estimate of the mean
	# This is a simplified version of formula (8) on page 4.
	middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)


	#print middle
	#print(middle)
	#print(Q-np.expand_dims(np.dot(P,pi.T),axis=1))
	er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))


	# Compute posterior estimate of the uncertainty in the mean
	# This is a simplified and combined version of formulas (9) and (15)
	posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
	#print(posteriorSigma)
	# Compute posterior weights based on uncertainty in mean
	w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
	# Compute lambda value
	# We solve for lambda from formula (17) page 7, rather than formula (18)
	# just because it is less to type, and we've already computed w*.
	lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)


	return [er, w, lmbda]



def printHello():
	print 'hello'



if __name__ == '__main__':


        rs = [[0.25,0.3,0.4, 0.3, 0.2, -0.1, -0.2], [0.1, 0.2, 0.3, -0.01, -0.2, 0.01, 0.02], [0.001, 0.02, -0.03, 0.05, -0.06, -0.07, 0.1]]
 	#risk ,returns, portfolio = efficient_frontier(rs)
	#print portfolio[0]
	#print np.cov(rs)
       	#print semivariance(rs[0])
	#print jensen(rs[0], rs[1], 0.02)
	#print sharp(rs[0], 0.02)
	#print sortino(rs[0], 0.02)
	#print tm(rs[0], rs[1], 0.02)
	#print ppw(rs[0], rs[1])


	weq = np.array([0.016,0.022,0.052,0.055,0.116,0.124,0.615])
	C = np.array([[ 1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
	      [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
	      [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
	      [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
	      [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
	      [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
	      [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
	Sigma = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
	refPi = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
	assets= {'Australia','Canada   ','France   ','Germany  ','Japan    ','UK       ','USA      '}	


	V = np.multiply(np.outer(Sigma, Sigma), C)

	delta = 2.5
	tau = 0.05


	tauV = tau * V
	P1 = np.array([0, 0, -.295, 1.00, 0, -.705, 0 ])
	P2 = np.array([1, 0, 0, 0, 0, -0.5, -0.5])  	
	Q1 = np.array([0.05])
	Q2 = np.array([0.06])
	P=np.array([P1, P2])
	Q=np.array([Q1, Q2]);

	#print P
	#print Q
	#print np.dot(np.dot(P,tauV),P.T)
	Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])	
	#print np.eye(Q.shape[0])
	#print Omega

	res = black_litterman(delta, weq, V, tau, P, Q, Omega)

	#print res


