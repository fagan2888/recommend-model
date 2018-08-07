#!/bin/bash
:<<!
# update valid stocks
python3 shell/roboadvisor.py sf factor_valid_update

# update stock factor exposre
python3 shell/roboadvisor.py sf factor_exposure_update

# update fund factor exposre
python3 shell/roboadvisor.py ff factor_exposure_update
!
# update single factor index
# python3 shell/roboadvisor.py markowitz --id MZ.FA0010 --new --start-date 2010-02-01
# python3 shell/roboadvisor.py markowitz --id MZ.FA0050 --new --start-date 2010-02-01
# python3 shell/roboadvisor.py markowitz --id MZ.FA0070 --new --start-date 2010-02-01
python3 shell/roboadvisor.py markowitz --id MZ.FA1010 --new --start-date 2010-02-01
:<<!
# update fund pool position
python3 shell/roboadvisor.py pool fund_factor_pool --id 11112001
python3 shell/roboadvisor.py pool fund_factor_pool --id 11112005
python3 shell/roboadvisor.py pool fund_factor_pool --id 11112007
python3 shell/roboadvisor.py pool fund_factor_pool --id 11113001
python3 shell/roboadvisor.py pool fund_factor_pool --id 11114009
python3 shell/roboadvisor.py pool fund_factor_pool --id 11114012
python3 shell/roboadvisor.py pool fund_factor_pool --id 11114018
python3 shell/roboadvisor.py pool fund_factor_pool --id 11114019

# update fund pool nav
python3 shell/roboadvisor.py pool nav --id 11112001
python3 shell/roboadvisor.py pool nav --id 11112005
python3 shell/roboadvisor.py pool nav --id 11112007
python3 shell/roboadvisor.py pool nav --id 11113001
python3 shell/roboadvisor.py pool nav --id 11114009
python3 shell/roboadvisor.py pool nav --id 11114012
python3 shell/roboadvisor.py pool nav --id 11114018
python3 shell/roboadvisor.py pool nav --id 11114019

# update fund pool turnvoer
python3 shell/roboadvisor.py pool turnover --id 11112001
python3 shell/roboadvisor.py pool turnover --id 11112005
python3 shell/roboadvisor.py pool turnover --id 11112007
python3 shell/roboadvisor.py pool turnover --id 11113001
python3 shell/roboadvisor.py pool turnover --id 11114009
python3 shell/roboadvisor.py pool turnover --id 11114012
python3 shell/roboadvisor.py pool turnover --id 11114018
python3 shell/roboadvisor.py pool turnover --id 11114019

# update composite asset
python3 shell/roboadvisor.py composite nav --asset MZ.FA0010
python3 shell/roboadvisor.py composite nav --asset MZ.FA0050
python3 shell/roboadvisor.py composite nav --asset MZ.FA0070
python3 shell/roboadvisor.py composite nav --asset MZ.FA1010
!
