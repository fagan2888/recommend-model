#coding=utf8


import pandas as pd
import numpy as np
import scipy.optimize
import MySQLdb


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


def obj_fun(x):
    p = 1
    for i in range(0, 10):
       p = p * ((((1 - 1.0 * i / 10) ** 0.5) * x) + 1 )
    return (p - 1.06) ** 2


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)

    gids = []
    for i in range(0, 10):
        gid = 80041010 + i
        gids.append(gid)

    gids_str =


    '''
    target_rs = []
    x = 0
    res = scipy.optimize.minimize(obj_fun, x, method='SLSQP')
    x = res.x[0]
    for i in range(0, 10):
        target_rs.append(((1 - 1.0 * i / 10) ** 0.5) * x)

    print target_rs
    '''
