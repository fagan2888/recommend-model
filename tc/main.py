# -*- coding: UTF-8 -*-

import momen
import tdsp
import datetime

if __name__ == "__main__":
    assets = ['HSI001', '000300', 'W00003', 'CI0022']
    assets = ['W00003']
    edate = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
    for ass in assets:
        print ass
        print edate
        gftd_ins = tdsp.TDSP(ass, '20040101', edate)
