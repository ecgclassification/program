import numpy as np
import matplotlib.pyplot as pl
import pybrain
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets            import SupervisedDataSet
from pybrain.tools.xml           import NetworkWriter
from pybrain.tools.xml           import NetworkReader
from sklearn.metrics             import accuracy_score

ds = SupervisedDataSet(30,1)
with open('datadctip.csv', 'rb') as f:      #importing the dataset.csv for inpt values "first 30"only
    results = []
    for line in f:
        words= line.split(',')
        results.append((words[:30]))
with open('datadctop.csv', 'rb') as f:      #importing the dataset.csv for target values "last 3"only
    result = []
    for line in f:
        words = line.split(',')
        result.append((words[:1]))
for i in range(0,150):
    ds.addSample((results[i]),result[i])
print len(ds),len(ds['input']),len(ds['target'])

trndata,partdata = ds.splitWithProportion (.60)
tstdata,validata = partdata.splitWithProportion(.50)

#trndata.indim,trndata.outdim,tstdata.indim, tstdata.outdim

# setting ANN
net = buildNetwork(30, 60, 1, bias=True,outclass=SoftmaxLayer )
trainer = BackpropTrainer(net,dataset=trndata,learningrate=0.01,momentum=0.5,verbose=False,weightdecay = 0.01)

trnerr,valerr = trainer.trainUntilConvergence(dataset=trndata,maxEpochs=50)
pl.plot(trnerr,'b',valerr,'r')
#pl.show()

out = net.activateOnDataset(tstdata)#.argmax(axis=1)
errroe=percentError(out, tstdata['input'])
#print "percent error",errroe
print out

#print "accuracy=",100-errroe
#acc = accuracy_score(validata,out)

out2=np.array([net.activate(x) for x, _ in tstdata])
#out2 = out2.argmax(axis=1)
print out2









