#!/bin/bash

if [ "$2" = "1" ];then
echo "copy markowitz model ..."
python shell/roboadvisor.py markowitz copy --src MZ.000020 --dst MZ.YJH010
#python shell/roboadvisor.py highlow copy --src HL.000020 --dst HL.YJH010
#python shell/roboadvisor.py portfolio copy --src PO.000020 --dst PO.YJH010
echo "copy markowitz model completion!"
fi

if [ "$1" = "1" ];then
echo "calculating macro view ..."
time python shell/roboadvisor.py mt macro_view_update
echo "macro view completion!"
fi

echo "makowitz allocation"
python shell/roboadvisor.py markowitz --id MZ.YJH010 --new
#python shell/roboadvisor.py highlow --id HL.YJH010 --new
#python shell/roboadvisor.py portfolio --id PO.YJH010 --new
