#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

from pymongo import MongoClient
from ipdb import set_trace

settings = {
    "ip":'127.0.0.1',   #ip
    "port":27017,           #端口
    "db_name" : "stock",    #数据库名字
    "set_name" : "test"   #集合名字
}

class MyMongoDB(object):
    def __init__(self, settings):
        self.conn = MongoClient(settings["ip"], settings["port"])
        self.db = self.conn[settings["db_name"]]
        self.my_set = self.db[settings["set_name"]]

    def insert(self, dic):
        # print("insert...")
        self.my_set.insert(dic)

    def insert_many(self, dic):
        # print("insert...")
        self.my_set.insert_many(dic)

    def update(self, dic, newdic):
        # print("update...")
        self.my_set.update_one(dic,newdic)

    def delete(self, dic):
        # print("delete...")
        self.my_set.delete_one(dic)

    def find(self, dic):
        # print("find...")
        data = self.my_set.find(dic)
        # for result in data:
        #     print(result)
        return data

def main():
    mongo = MyMongoDB(settings)

    dics=[
        {"name":"zhangsan","age":18},
        {"name":"zhangsan","age":30},
        {"name":"zhangsan","age":50},
    ]
    # for dic in dics:
    #     mongo.insert(dic)
    mongo.insert_many(dics)

    mongo.find({"name":"zhangsan"})

    mongo.update({"name":"zhangsan"},{"$set":{"age":"25"}})
    mongo.find({"name":"zhangsan"})

    mongo.delete({"name":"zhangsan"})
    mongo.find({"name":"zhangsan"})

if __name__ == "__main__":
    main()








