
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl
# %matplotlib inline

def perceptron(xn, yn, max_iter=1000, w=np.zeros(3)):
    '''
        A very simple implementation of the perceptron algorithm for two dimensional data.

        Given points (x, y) with x in R^{2} and y in {-1, 1}, the perceptron learning algorithm
        searchs for the best line the separates the data points according to the difference classes defind
        in y.
    '''
    # get number of data in xn
    N = xn.shape[0]

    # Separating curve
    # turn array value to sign (-1, 0, 1) for a linear function
    f = lambda x: np.sign(w[0]+w[1]*x[0]+w[2]*x[1])

    for _ in xrange(max_iter):
        # random sample those points. i would be interger and between 0 to N
        i = nr.randint(N)
        # print i, xn[i, 0], xn[i, 1], f(xn[i,:]), yn[i]
        # If not classified correctly, adjust the line to account for that point
        if (yn[i] != f(xn[i,:])):
            # the first weight is effectively the bias
            w[0] = w[1] + yn[i]
            w[1] = w[1] + yn[i] * xn[i, 0]
            w[2] = w[2] + yn[i] * xn[i, 1]
    return w

# This notebook is based on an excellent tutorial by Kostis Gourgoulias (http://kgourgou.me/)

# Specify size of plot
pl.rcParams['figure.figsize'] = (12.0, 10.0)

# Generate some points
N = 100
# N points two dimensions
xn = nr.rand(N, 2)

# Generate date array from 0 to 1, default array size is 50 .
x = np.linspace(0, 1)

# Pick a line
a, b = 0.8, 0.2
f = lambda x : a*x + b

fig = pl.figure()
# Get current axis
figa = pl.gca()
# plot random points
# pl.plot(xn[:, 0], xn[:, 1], 'bo')
# plot line
pl.plot(x, f(x), 'r')

# points Separate flag
yn = np.zeros([N, 1])

for i in xrange(N):
    # Compare isn't the point above or below the line.
    if(f(xn[i, 0]) > xn[i, 1]):
        yn[i] = 1
        # pl.plot(xn[i, 0], xn[i, 1], 'go')
    else:
        yn[i] = -1
w = perceptron(xn, yn)
# turning ax + by + c = 0 form to y = ax + b
bnew = -w[0]/w[2]   # bnew = -c/b
anew = -w[1]/w[2]   # anew = -a/b
y = lambda x: anew * x + bnew
# compute colors for the points
sep_color = (yn+1)/2.0
# pl.figure()
# figa
pl.scatter(xn[:,0], xn[:,1], c=sep_color, s=30)
pl.plot(x,y(x),'b--',label='Line from perceptron implementation.')
pl.plot(x,f(x),'r',label='Original line.')
pl.legend()
# pl.legend(['Above','Separator','Below'],loc=0)
pl.title('Selected points with their separating line.')
pl.show()
