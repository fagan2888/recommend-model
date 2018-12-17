# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:22:03 2018

@author: yshlm
"""

from functools import partial
import pylab
from MCMC import *
 # MCMC and Gibbs Sampling, by Walsh, 2004, p.8
 # proposal dist. is uniform (symmetric) -> metropolis
if __name__ == '__main__':
# 
#    f = partial(inv_chi_sq, n = 5, a = 4)
#    prop = partial(uni_prop, frm=0, to = 100)
#    smpls = run_chain(metropolis, f, prop, 1, 50000)
#    pylab.plot(smpls[0])


     # MCMC and Gibbs Sampling, Walsh, p. 9
    f = partial(inv_chi_sq, n = 5, a = 4)
    prop = partial(chi_sq, n=1)
    smpls = run_chain(metropolis, f, prop, 1, 50000)
    pylab.plot(smpls[0])

    f = partial(inv_chi_sq, n = 5, a = 1)
    prop = partial(chi_sq, n=1)
    smpls = run_chain(metropolis, f, prop, 1, 50000)
    pylab.plot(smpls[0])


