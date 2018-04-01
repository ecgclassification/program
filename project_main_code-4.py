from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(30, 1)
with open('dataset.csv', 'rb') as f:
    results = []
    for line in f:
      words = line.split(',')
      results.append((words[:30]))
with open('dataset.csv', 'rb') as f:
    result = []
    for line in f:
      words = line.split(',')
      result.append((words[30]))
for i in range(0,1200): 
 ds.addSample ((results[i]),result[i])
 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader
net = NetworkReader.readFrom('filename.xml')
print('test output')
print ds
import numpy as np

a = [ -125.3599173508 ,  -80.9561499902 ,  -16.0020082484 ,  223.1696856395 ,  -60.981089846 ,  -159.435720846 ,  34.4359393929 ,  174.070154076 ,  -7.1801617452 ,  -145.7213850274 ,  2.9203769057 ,  99.0630057126 ,  -21.1487240936 ,  -90.0975609849 ,  23.341864798 ,  79.1595173465 ,  -13.9341115137 ,  -67.6949931498 ,  16.2199767871 ,  57.2245028304 ,  -5.2730858721 ,  -56.0025453955 ,  14.1840898814 ,  43.4259339039 ,  -9.3798037823 ,  -37.0155984412 ,  8.89659141 ,  25.4182968278 ,  -7.2060022901 ,  -26.6391150877 ]
 
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


print result
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

