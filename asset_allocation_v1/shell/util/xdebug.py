#coding=utf8


# import string
# from datetime import datetime, timedelta
# import os
import sys
# import logging

def dd(*args, **kwargs):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    if len(args) > 0 and len(kwargs) > 0:
        print args, kwargs
        sys.exit(0)

    if len(args) > 0:
        print args
    elif len(kwargs) > 0:
        print kwargs

    sys.exit(0)
