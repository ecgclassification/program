import pybrain
from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(30, 3)
with open('feature.csv', 'rb') as f:
    results = []
    readCSV = csv.reader(csvfile, delimiter=',')
    x=list(readCSV)
    x = normalize(x,axis=0,norm='max')
    x_normed =(x-x.min(0))/x.ptp(0)
       
    results.append((x_normed[:30]))
              
        
with open('feature1.csv', 'rb') as f:
    result = []
    for line in f:
        words = line.split(',')
        result.append((words[:3]))
              
for i in range(0,10):
    ds.addSample((results[i]),result[i])

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader
net = NetworkReader.readFrom('filename.xml')
print('test output')
print ds
import numpy as np

a = [ 2.97686000e+05  -1.30948746e+04  -1.17229528e+04  -1.88423173e+04
    1.07343441e+04  -1.02415470e+04  -1.34167316e+04  -2.03197752e+03
    9.23824486e+03   9.91710405e+01  -9.91114408e+03   1.24103440e+03
    8.64959897e+03  -3.57153259e+02  -7.47899732e+03   1.90694506e+03
    6.99316858e+03  -1.34982426e+03  -6.46878296e+03   1.68821277e+03
    4.89234077e+03  -1.46774964e+03  -3.88744951e+03   1.94510792e+03
    2.95137205e+03  -1.53892885e+03  -2.08624436e+03   1.36426126e+03
    9.86969032e+02  -7.80173972e+02 ]

result = net.activate(a)
#A = np.array(a)
#npad = 200 - len(A)
#a = np.pad(A, pad_width=npad, mode='constant', constant_values=0)[npad:]
#p = np.fft.ifft(a)
#A = np.array(p)
#B = A[:100]
#print B
import matplotlib.pyplot as plt
import math
T=0.002
from scipy.fftpack import dct
from scipy.fftpack import idct
B = idct(a)
plt.plot(a)
plt.show()


print result
'''
if result > 0.5:
    result = ('normal')
 print result
 plt.plot(B)
 plt.title('normal')
 plt.show()
 
else:
 result = ('abnormal')
 print result
 plt.plot(B)
 plt.title('abnormal')    
 plt.show()
'''
