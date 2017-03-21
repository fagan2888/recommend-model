# -*- coding: UTF-8 -*-

import momen
import tdsp
import datetime
import gftd

if __name__ == "__main__":
    assets = ['HSI001', '000300', 'W00003', 'CI0022']
    assets = ['CI0022']
    edate = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
    sdate = "20050101"
    file_handle = "000905_data.csv"
    for ass in assets:
        print ass
        print edate
        # gftd_ins = tdsp.TDSP(ass, '20040101', edate)
        gftd_ins = gftd.GFTD(ass, sdate, edate, file_handle)
