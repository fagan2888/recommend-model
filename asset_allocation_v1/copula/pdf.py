#!/usr/bin/python
# coding=utf-8

from numpy import random, histogram2d, diff, arange, meshgrid, vectorize, meshgrid
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D  
from ipdb import set_trace


'''
# Generate sample data
set_trace()
arr = arange(10.0)
n = 10000
x = random.randn(n)
y = -x + random.randn(n)

# bin
nbins = 100
H, xedges, yedges = histogram2d(x, y, bins=nbins)

# Figure out centers of bins
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2

xcenters = centers(xedges)
ycenters = centers(yedges)

# Construct interpolator
pdf = interp2d(xcenters, ycenters, H)
# set_trace()

# test
# plt.pcolor(xedges, yedges, pdf(xedges, yedges))
# plt.show()

from numpy import meshgrid, vectorize

def position(edges, value):
    return int((value - edges[0])/diff(edges[:2]))

@vectorize
def pdf2(x, y):
    return H[position(yedges, y), position(xedges, x)]

# test - note we need the meshgrid here to get the right shapes
xx, yy = meshgrid(xcenters, ycenters)
plt.pcolor(xedges, yedges, pdf2(xx, yy))
plt.show()
'''

def PDF(x, y, nbins = None, plot = False):
    if nbins is None:
        nbins = len(x)
    H, xedges, yedges = histogram2d(x, y, bins=nbins)
    xcenters =  centers(xedges)
    ycenters =  centers(yedges)
    pdf_func = interp2d(xcenters, ycenters, H, kind = 'cubic')
    if plot:
        fig = plt.figure()
        # ax = fig.add_subplot(111) 
        # set_trace()
        pdf_value = pdf_func(xedges, yedges)
        # ax.pcolor(xedges, yedges, pdf_value)

        # xx, yy = meshgrid(xedges, yedges)
        xx, yy = meshgrid(xcenters, ycenters)
        ax = Axes3D(fig)
        # ax.plot_surface(xx, yy, pdf_value, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        # ax.contourf(xx, yy, pdf_value, zdir='z', offset=-2)
        ax.plot_surface(xx, yy, H, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        ax.contourf(xx, yy, pdf_value, zdir='z', offset=-2)
        ax.set_zlim(0,1)  
        plt.savefig('test.pdf')

        # plt.show()

    return pdf_func
    # return 0

def centers(edges):
    return edges[:-1] + diff(edges[:2])/2


if __name__ == '__main__':

    n = 10000
    x = random.randn(n)
    y = -2*x + random.randn(n)
    pdf = PDF(x, y, 100, plot = True)