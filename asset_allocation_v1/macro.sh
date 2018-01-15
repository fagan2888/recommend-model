#!/bin/bash

if [ "$1" = "1" ];then
echo "calculating macro view ..."
time python shell/macro_view.py
echo "macro view completion!"
fi
echo "makowitz allocation"
python shell/roboadvisor.py markowitz --id MZ.YJH010 --new
python shell/roboadvisor.py highlow --id HL.YJH010 --new
python shell/roboadvisor.py portfolio --id PO.YJH010 --new
