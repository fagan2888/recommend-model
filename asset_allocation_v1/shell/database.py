#coding=utf8


import string
import MySQLdb
import config
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys
import logging
sys.path.append('shell')
import Const

from sqlalchemy import *


from dateutil.parser import parse

logger = logging.getLogger(__name__)

uris = {
    'asset': config.db_asset_uri,
    'base': config.db_base_uri,
}

connections = {}

def connection(key):
    if key in connections:
        return connections[key]

    if key in uris:
        uri = uris[key]
        con = create_engine(uri)
        connections[key] = con
        return con

    return None

    
