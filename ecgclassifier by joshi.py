
import xlrd
import numpy as np
from sklearn import tree#svm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
workbook= xlrd.open_workbook('DATADCT.xls')
sheet=workbook.sheet_by_index(1)
sheet1=workbook.sheet_by_index(2)
'''
train =np.zeros(400,30)
for j in range(0,400):

    train=[sheet.cell_value(j,i) for i in range (0,30)]

print train
'''
num_rows=400#sheet.nrows
num_cols=30#sheet.ncols

train=[]
label=[]

for curr_row in range(0,num_rows,1):
    row_data=[]
   
    for curr_col in range(0,num_cols,1):

        data= sheet.cell_value(curr_row,curr_col)
        
        row_data.append(data)
    
    train.append(row_data)
    
col=1    
for curr_row in range(0,num_rows,1):
   
    row_data1=[]
    for curr_col in range(0,col,1):

       
        data1= sheet1.cell_value(curr_row,curr_col)
       
        row_data1.append(data1)
   
    label.append(row_data1)
  
train_feats,test_feats,train_labels,test_labels=tts(train,label,test_size=0.2)
#print train_feats
#print train_labels
train_labels=np.ravel(train_labels)
test_labels=np.ravel(test_labels)
print test_labels

clf=tree.DecisionTreeClassifier()
#clf=svm.SVC(kernel='poly')

# train
clf.fit(train_feats,train_labels)

#prediction
predictions=clf.predict(test_feats)
print predictions
'''
score=0
for i in range (len(predictions)):
    if predictions[i]==test_labels[i]:
        score+=1
print score       
'''
z=accuracy_score(test_labels,predictions)
print z
