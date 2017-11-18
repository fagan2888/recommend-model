from rpy2 import robjects
from rpy2 import rinterface as ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, data
import os
import numpy as np
numpy2ri.activate()

Vars = importr('vars')
datasets = importr('datasets')
#mtcars = data(datasets).fetch('mtcars')['mtcars']
#vs = Vars.VARselect(mtcars, 5, 'const')
#model = Vars.VAR(mtcars, 2, 'const')
data = np.column_stack([np.arange(100), np.random.randn(100)])
#print data
#os._exit(0)


robjects.r('''
        train <- function(data) {
            library(vars)
            data(Canada)
            var2c = VAR(Canada, p = 2, type = 'const')
            #summary(model)
            #plot(model)

            #var2c_ser = restrict(var2c, method = 'ser', thresh = 2)
            #print(var2c_ser$restrictions)
            #summary(var2c_ser)

            #var_f10 = predict(var2c, n.ahead = 10, ci = 0.95)
            #print(var_f10)
            amat = diag(4)
            diag(amat) = NA
            amat[1,2] = NA
            amat[1,3] = NA
            amat[3,2] = NA
            amat[4,1] = NA

            svar2c = SVAR(var2c, Amat = amat, Bmat = NULL, hessian = TRUE, method = 'BFGS')
            svar2c_ira = irf(svar2c, impulse = 'rw', response = c('e', 'U'), boot = FALSE)
            print(svar2c_ira)
            #print(Canada)
            #print(svar2c)
        }
        ''')

r_train = robjects.r['train']
r_train(data)
#res = r_train(data)
#print res
