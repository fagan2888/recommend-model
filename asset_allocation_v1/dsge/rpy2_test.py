from rpy2 import robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np
numpy2ri.activate()
stats = importr('stats')
base = importr('base')

robjects.r('''
        f <- function(r, verbose = FALSE){
            if (verbose){
                cat("I am calling f().\n")
                }
                return(2*pi*r)
           }
            ''')

robjects.r('''
        f2 <- function(r, verbose = FALSE){
            if (verbose){
                cat("I am calling f2().\n")
                }
                return(pi*r)
           }
            ''')

robjects.r('''
        f3 <- function(arr, verbose = FALSE){
            print(arr[1,])
            }
            ''')

r_f = robjects.r['f']
r_f2 = robjects.r['f2']
res = r_f(4, True)
res2 = r_f2(4, True)
print res
print res2

np.random.seed(1)
arr = np.random.randn(4,4)
r_f3 = robjects.r['f3']
r_f3(arr)
numpy2ri.deactivate()
