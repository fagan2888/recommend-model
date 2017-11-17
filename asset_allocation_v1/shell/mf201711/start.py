#coding=utf-8
import os,sys
import time
import subprocess
import shutil
import pandas as pd

if_ST = sys.argv[1]
groupnum = sys.argv[2]
holdlimit = sys.argv[3]
sizelimit = sys.argv[4]
alphalimit = sys.argv[5]
alphakind = sys.argv[6]
try:
    method = sys.argv[7]
except:
    method = 'cross'

join = '_'.join(['FINAL',if_ST,groupnum,holdlimit,sizelimit,alphalimit,alphakind])
try:
    shutil.rmtree('copys/%s' %join)
except:
    pass
shutil.copytree('copys/init','copys/%s' %join)

print os.system('python copys/%s/new_validcode.py %s' %(join,if_ST) )
print os.system('python copys/%s/new_normalizer.py' %join )

print os.system('python copys/%s/new_singleFactorTest.py %s' %(join,groupnum) )
print os.system('python copys/%s/factorTestResult/layeredconcat.py' %join )
print os.system('python copys/%s/factorIndex.py %s' %(join,groupnum) )
print os.system('python copys/%s/fundFactorValue.py' %join )

print os.system('python copys/%s/new_funddata/validfund.py %s %s %s %s' %(join,holdlimit,sizelimit,alphalimit,alphakind) )
print os.system('python copys/%s/new_fundPool-Terminal.py %s' %(join,method) )
print os.system('python copys/%s/netprice-Terminal.py' %join )
print os.system('python copys/%s/factorTestResult/alloMethod_corrMean/effblockist.py' %join )
print os.system('python copys/%s/factorTestResult/alloMethod_corrMean/fundpricelookback.py' %join )
stockresult = pd.read_csv('copys/%s/factorTestResult/alloMethod_corrMean/stockresult.csv' %join ,index_col=0,header=None)
fundresult = pd.read_csv('copys/%s/factorTestResult/alloMethod_corrMean/fundresult.csv' %join ,index_col=0,header=None)
sumresult = pd.DataFrame([[if_ST,groupnum,holdlimit,sizelimit,alphalimit,alphakind]+list(stockresult.values.flat)+list(fundresult.values.flat)],
                         columns=['if_ST','groupnum','holdlimit','sizelimit','alphalimit','alphakind']+list(stockresult.index)+list(fundresult.index))
sumresult.to_csv('sumresult/%s.csv' %(join) )
