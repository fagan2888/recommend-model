#!/bin/bash
python3 /home/yaojiahui/recommend_model2/recommend_model/asset_allocation_v2/shell/roboadvisor.py sf factor_valid_update
python3 /home/yaojiahui/recommend_model2/recommend_model/asset_allocation_v2/shell/roboadvisor.py sf factor_exposure_update
python3 /home/yaojiahui/recommend_model2/recommend_model/asset_allocation_v2/shell/roboadvisor.py ff factor_exposure_update
bash factor_index.sh
bash fund_pool_valid.sh
bash import_composite.sh
