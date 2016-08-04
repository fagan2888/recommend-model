#coding=utf8

import sys
import string
import numpy as np
import math
from scipy.stats import norm


def down_sd(rs):
    array = np.array(rs)
    mean  = array.mean()
    n = 0
    var_sum = 0
    for r in rs:
        if(r < mean):
            var_sum = var_sum + (r - mean) * (r - mean)
            n = n + 1
    var = var_sum / n
    return math.sqrt(var)
            

f = open(sys.argv[1], 'r')


lines = f.readlines()


out = open('names','w')
for line in lines:
    vec = line.split(',')
    name = vec[0].strip()
    values = vec[1:len(vec) - 1]
    #print values        
    
    price_change_ratio = []            
    for i in range(1, len(values)):
        price_change_ratio.append(string.atof(values[i]) / string.atof(values[i - 1]) - 1)

    parray = np.array(price_change_ratio)
    sharp = parray.mean() / parray.std()
    d_sd =    down_sd(price_change_ratio)
    sortino_ratio = parray.mean() / d_sd    
    valueAtRisk = norm.ppf(0.05, parray.mean(), parray.std())                
    print sharp, d_sd, sortino_ratio, valueAtRisk
    out.write(name + "\n")
out.close()
                                                                    
        #print parray.std()
        #print 
