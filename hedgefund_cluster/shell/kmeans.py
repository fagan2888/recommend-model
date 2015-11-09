#coding=utf8


from sklearn.cluster import KMeans


clf = KMeans(n_clusters=9)
s = clf.fit(feature)
print s
