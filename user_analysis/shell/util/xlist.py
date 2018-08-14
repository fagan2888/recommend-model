#coding=utf8


# import string
# from datetime import datetime, timedelta
# import os
# import sys
# import logging
import pdb

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # pdb.set_trace()
    for i in range(0, len(l), n):
        yield l[i:i + n]
