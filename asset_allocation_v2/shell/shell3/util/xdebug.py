#coding=utf8


# import string
# from datetime import datetime, timedelta
# import os
import sys
import pandas as pd
# import logging

def dd(*args, **kwargs):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    if len(args) > 0 and len(kwargs) > 0:
        print("")
        for arg in args:
            print(arg)
        for k, v in list(kwargs.items()):
            print((k, v))
        sys.exit(0)

    if len(args) > 0:
        print("")
        for arg in args:
            print(arg)
    elif len(kwargs) > 0:
        for k, v in list(kwargs.items()):
            if isinstance(v, pd.DataFrame):
                print((k, ":", type(v)))
                print(v)
            else:
                print((k, ":",  v))

    sys.exit(0)
