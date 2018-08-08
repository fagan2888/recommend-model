#!/bin/bash
# update valid stocks
/home/finance/anaconda3/bin/python shell/roboadvisor.py sf factor_valid_update

# update stock factor exposre
/home/finance/anaconda3/bin/python shell/roboadvisor.py sf factor_exposure_update

# update fund factor exposre
/home/finance/anaconda3/bin/python shell/roboadvisor.py ff factor_exposure_update

# update single factor index
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0010 --new --start-date 2010-02-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0050 --new --start-date 2010-02-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0070 --new --start-date 2010-02-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1010 --new --start-date 2010-02-01

# update fund pool position
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11112001
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11112005
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11112007
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11113001
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11114009
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11114012
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11114018
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11114019

# update fund pool nav
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11112001
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11112005
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11112007
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11113001
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11114009
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11114012
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11114018
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11114019

# update fund pool turnvoer
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11112001
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11112005
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11112007
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11113001
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11114009
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11114012
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11114018
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11114019

# update composite asset
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0010
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0050
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0070
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1010
