#!/bin/bash

# update valid stocks
/home/finance/anaconda3/bin/python shell/roboadvisor.py sf factor_valid_update

# update stock factor exposre
/home/finance/anaconda3/bin/python shell/roboadvisor.py sf factor_exposure_update

# update fund factor exposre
/home/finance/anaconda3/bin/python shell/roboadvisor.py ff factor_exposure_update

# update single factor index
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0010 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0020 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0030 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0040 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0050 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0060 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0070 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0080 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA0090 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1010 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1020 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1030 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1040 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1050 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1060 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1070 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1080 --new --start-date 2008-01-01
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.FA1090 --new --start-date 2008-01-01

# update fund pool position
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11130100
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11130500
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11130700
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11140100
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11150900
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11151200
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11151800
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool fund_factor_pool --id 11151900

# update fund pool nav
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11130100
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11130500
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11130700
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11140100
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11150900
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11151200
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11151800
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool nav --id 11151900

# update fund pool turnvoer
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11130100
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11130500
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11130700
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11140100
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11150900
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11151200
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11151800
/home/finance/anaconda3/bin/python shell/roboadvisor.py pool turnover --id 11151900

# update composite asset
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0010
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0020
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0030
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0040
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0050
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0060
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0070
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0080
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA0090
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1010
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1020
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1030
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1040
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1050
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1060
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1070
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1080
/home/finance/anaconda3/bin/python shell/roboadvisor.py composite nav --asset MZ.FA1090

# update valid factor
/home/finance/anaconda3/bin/python shell/roboadvisor.py markowitz --id MZ.MF0010 --new
