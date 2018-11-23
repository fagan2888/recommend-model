#!/bin/bash

historyTime_1m=$(date "+%Y-%m-%d" -d '1 month ago')
historyTime_3m=$(date "+%Y-%m-%d" -d '3 month ago')
historyTime_6m=$(date "+%Y-%m-%d" -d '6 month ago')
historyTime_12m=$(date "+%Y-%m-%d" -d '12 month ago')

python shell/roboadvisor.py markowitz --id MZ.CB0010 --new --start-date $historyTime_1m
python shell/roboadvisor.py highlow --id HL.CB0010 --new
python shell/roboadvisor.py portfolio --id PO.CB0010 --new

python shell/roboadvisor.py markowitz --id MZ.CB0020 --new --start-date $historyTime_3m
python shell/roboadvisor.py highlow --id HL.CB0020 --new
python shell/roboadvisor.py portfolio --id PO.CB0020 --new

python shell/roboadvisor.py markowitz --id MZ.CB0030 --new --start-date $historyTime_6m
python shell/roboadvisor.py highlow --id HL.CB0030 --new
python shell/roboadvisor.py portfolio --id PO.CB0030 --new

python shell/roboadvisor.py markowitz --id MZ.CB0040 --new --start-date $historyTime_12m
python shell/roboadvisor.py highlow --id HL.CB0040 --new
python shell/roboadvisor.py portfolio --id PO.CB0040 --new

python shell/roboadvisor.py composite nav --asset 20301
python shell/roboadvisor.py composite nav --asset 20302
python shell/roboadvisor.py composite nav --asset 20303
python shell/roboadvisor.py composite nav --asset 20304
