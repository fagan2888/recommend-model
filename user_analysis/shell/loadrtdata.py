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
import sys

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

    def search(self, es, index, doc, query, t='10m'):
        try:
            all_doc = es.search(index=index, doc_type=doc, body=query)
            return all_doc['hits']['hits']
        except:
            return []

    def load_access_data(self):
        _from = 0
        _size = 10
        lines = []
        while True:
            #query = {"query":{"range":{"c_time":{"lt":"now","gt":"now-24h"}}},
            #        "sort":[{"c_time":"desc"}],
            #        "from":_from,
            #        "size":_size,
            #        }
            query = {"query":{"match_all":{}},
                    "sort":[{"c_time":"desc"}],
                    "from":_from,
                    "size":_size,
                    }

            res = self.search(self.es, self.index, self.doc, query)
            #print(len(res))
            _from += _size
            if len(res) == 0:
                break


        print(len(lines))

    def run(self):
        self.load_access_data()

if __name__=="__main__":
    AccessData().run()
