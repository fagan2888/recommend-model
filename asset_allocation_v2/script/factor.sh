#!/bin/bash
python shell/roboadvisor.py sf factor_valid_update
python shell/roboadvisor.py sf factor_exposure_update
python shell/roboadvisor.py ff factor_exposure_update
# bash factor_index.sh
bash fund_pool_pos_style.sh
bash fund_pool_style.sh
bash fund_pool_pos_ind.sh
bash fund_pool_ind.sh
# bash import_composite.sh
