#!/bin/bash
pydir=$1
fname=$2
hnum=$3
lnum=$4
debug="n"
echo $fname, $hnum, $lnum, $debug,"$USER"
cd $pydir
python ./shell/RiskAssetAllocation.py $fname $hnum $lnum $debug
