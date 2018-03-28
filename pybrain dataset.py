
import pybrain
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader
import matplotlib.pyplot as plt

ds = SupervisedDataSet(30, 3)
with open('feature.csv', 'rb') as f:
    results = []
    for line in f:
        words = line.split(',')
        results.append((words[:30]))
with open('feature1.csv', 'rb') as f:
    result = []
    for line in f:
        words = line.split(',')
        result.append((words[:3]))
for i in range(0,10):
    ds.addSample ((results[i]),result[i])

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt
net = buildNetwork(30, 60, 3, bias=True, hiddenclass=TanhLayer)
t = BackpropTrainer(net,ds,learningrate=0.01,momentum=0.5,verbose=True)
print t

# gives error
t.train()


for inpt, target in ds:
     print inpt,target
     print('test output')
     t.trainOnDataset(ds,120)
     t.testOnData(verbose=True)
print ds
a = [593.5597577,-819.2664339,-697.0721867,703.032348,400.1569157,-90.83375558,-372.4963064,65.59649783,86.22512954,-5.47130456,12.75411165,65.17186203,-60.36621132,-87.65988451,67.48946006,86.15059551,-67.1710396,-80.19250411,70.08677663,86.2910155,-60.42018586,-88.51686811,39.86637584,57.22098767,-18.20147468,-51.38013362,10.27697365,29.13768727,-9.481942915,-18.10302233]
result = net.activate(a)
print result
NetworkWriter.writeToFile(net, 'filename.csv')
net = NetworkReader.readFrom('filename.xml')


