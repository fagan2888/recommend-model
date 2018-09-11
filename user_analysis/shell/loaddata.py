# coding: utf-8
__author__ = 'luzhi'

import time
import re
import json
import time
import traceback
import logging
import datetime
#import urlparse

#from urllib import parse
from elasticsearch import helpers
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch

class AccessData(object):
    def __init__(self):
        #ES_CONFIG = {
        #    "host":"10.111.66.51", 
        #    "port":"9200", 
        #    "user":"luzhi", 
        #    "passwd":"123456a" 
        #}
        #self.es = Elasticsearch([ES_CONFIG['host']], http_auth=(ES_CONFIG['user'], ES_CONFIG['passwd']), port=ES_CONFIG['port']) 
        #self.date = datetime.datetime.now().strftime("%y-%m-%d") 
        #self.index = 'tupu-tjaccess.%s' % self.date 
        #self.doc = "tjaccess_doc" 
        #self.points = script_points.ScriptPoints() 
        #self.dids = {} 
        self.es = Elasticsearch(['10.111.66.91'], http_auth=('elastic', 'eSl&aPs5t3i1c'), port='9200') 
        self.date = datetime.datetime.now().strftime("%y-%m-%d")
        self.index = 'tupu.access.*'
        self.doc = "access_doc"
    def scan(self, es, index, doc, query, t='10m'):
        try:
            recs = []
            all_doc = helpers.scan(es, query, t, index=index, doc_type=doc, raise_on_error=True, preserve_order=False, request_timeout=1200)
            return all_doc
        except:
            return []

    def load_access_data(self):
        flag = True
        query = {"query":{"match_all":{}}}
        while flag:
            res = self.scan(self.es, self.index, self.doc, query)
            f = open('data.txt','a')
            for x in res:
                f.write(str(x))
                f.write("\n")
            f.close
            break

    def run(self):
        self.load_access_data()

if __name__=="__main__":
    AccessData().run()


