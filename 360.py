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
from sklearn.metrics             import confusion_matrix

ds = ClassificationDataSet(30, 3, nb_classes=3)
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

    for i in range(0,10):
        ds.addSample ((results[i]),result[i]) #creating dataset for ANN 


trndata,partdata = ds.splitWithProportion (.60)
tstdata,validata = partdata.splitWithProportion(.50)
'''
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
validata._convertToOneOfMany()
'''

#trndata.indim,trndata.outdim,tstdata.indim, tstdata.outdim

# setting ANN

net = buildNetwork(30,40,3, bias=True,outclass=SoftmaxLayer)
trainer=BackpropTrainer(net,dataset=trndata,momentum=0.5,learningrate=0.01,verbose=False,weightdecay=0.01)
trnerr,valerr = trainer.trainUntilConvergence(dataset=trndata,maxEpochs=50)
pl.plot(trnerr,'b',valerr,'r')
#pl.show()
trainer.trainOnDataset(trndata,50)
out = net.activateOnDataset(tstdata).argmax(axis =1)
err=percentError(out, tstdata['class'])
#print "percent error",errroe
print err
'''
out=net.activateOnDataset(tstdata)
print out
out =out.argmax(axis=1)
output =np.array([net.activate(x) for x, _ in validata])
output = output.argmax(axis=1)
print output
err1=percentError(output, validata['class'])
print err1
##########################################################################################

fnn = buildNetwork(30, 60, 3, bias=True,outclass=SoftmaxLayer )
tr = BackpropTrainer(fnn,dataset=trndata,momentum=0.5,verbose=False,weightdecay=0.3,learningrate=0.3)
tr.trainOnDataset(trndata,1000)

out=fnn.activateOnDataset((tstdata))#.argmax(axis =1))
per=percentError(out,tstdata['class'])
                          
out1=fnn.activateOnDataset((validata))#.argmax(axis =1))
err2=percentError(out,validata['class'])

print err2

confusion_matric = confusion_matrix(out,out1)
print confusion_matric
'''
