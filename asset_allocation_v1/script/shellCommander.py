import os
import time
from globalvalue import *

lastid = str(30200+idrange[len(idrange)-1])

com99 = 'python script/99_deleteAllocation.py'
com0 = 'python script/0_assetsignup.py'
com33 = 'python script/3.3_fundPool.py'
com42 = 'python script/4.2_factorAllocation.py'
com5 = 'python script/5_fundAllocation.py 20130101'
comr = 'python shell/roboadvisor.py composite nav --asset $(seq -s , 30200 '+lastid+')'
comcpt = 'python script/cptShifter.py'

coms = [com99,com33,com42,com0,com5,comr,comcpt]
for com in coms:
    print os.popen(com).read()
    time.sleep(1)
    
