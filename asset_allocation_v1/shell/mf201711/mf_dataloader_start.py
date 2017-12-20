#coding=utf-8
import sys
import subprocess

splitnum = int(sys.argv[1])
for splitloc in range(0,splitnum):
    print subprocess.Popen('/home/huyang/anaconda2/bin/python mf_dataloader.py %s %s' %(str(splitnum),str(splitloc)),shell=True)
