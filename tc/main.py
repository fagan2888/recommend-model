# -*- coding: utf-8 -*-

import momen
import gftd
import datetime

if __name__ == "__main__":
    assets = ['HSI001', '000300', 'W00003', 'CI0022', '000905']
    assets = ['000300']
    edate = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
    for ass in assets:
        print ass
        print edate
        gftd_ins = gftd.GFTD(ass, '20100101', edate)
