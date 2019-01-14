# coding: utf-8

__author__ = 'jiaoyang'

import time
import re
import json
import time
import traceback
import logging
import datetime

from elasticsearch import helpers
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import sys

class ESData(object):

    def __init__(self):
        self.es = Elasticsearch(['10.111.66.91'], http_auth=('elastic', 'eSl&aPs5t3i1c'), port='9200') 
        self.date = datetime.datetime.now().strftime("%y-%m-%d")
        self.index = 'tupu.access.*'
        self.doc = "access_doc"


    def search(self, es, index, doc, query, t='10m'):
        try:
            all_doc = es.search(index=index, doc_type=doc, body=query)
            return all_doc['hits']['hits']
        except:
            return []

    def count(self, es, index, doc, query):
        try:
           all_doc = es.search(index=index, doc_type=doc, body=query)
           return all_doc['hits']['total']
        except:
            return 0

    def scan(self, es, index, doc, query, t='10m'):
        try:
            recs = []
            all_doc = helpers.scan(es, query, t, index=index, doc_type=doc, raise_on_error=False, preserve_order=False, request_timeout=1200)
            return all_doc
        except:
            return []

    def load_access_data(self, query):

        lines = []
        res = self.scan(self.es, self.index, self.doc, query)
        for x in res:
            lines.append(x)

        return lines
