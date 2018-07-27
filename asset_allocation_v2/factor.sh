#!/bin/bash
/home/yaojiahui/recommend_model2/recommend_model/asset_allocation_v2/python3 shell/roboadvisor.py sf factor_valid_update
/home/yaojiahui/recommend_model2/recommend_model/asset_allocation_v2/python3 shell/roboadvisor.py sf factor_exposure_update
/home/yaojiahui/recommend_model2/recommend_model/asset_allocation_v2/python3 shell/roboadvisor.py ff factor_exposure_update
bash factor_index.sh
bash fund_pool_pos_style.sh
bash fund_pool_style.sh
bash fund_pool_pos_ind.sh
bash fund_pool_ind.sh
bash import_composite.sh
