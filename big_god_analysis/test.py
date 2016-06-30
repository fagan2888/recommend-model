#coding=utf8
a = {}
b = {}
b[0] = {"a":"apple", "b":"bana"}
a[0] = b.copy()
b[0] = {"a":"pple", "b":"bana"}
print a
