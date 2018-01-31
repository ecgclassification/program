import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from math import exp

from random import random


file_location = r'I:\Git Hub\ECG clssi\ECG code\training_and_testing.xlsx'
workbook = xlrd.open_workbook(file_location)
#input matrix
ip = np.empty((200,30))
file_location = r'training_and_testing.xlsx'
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
for j in range (0,20):
  xi= [first_sheet.cell_value(j,i) for i in range (30)]
  #print xi
  ip[j,:]=(xi)
#isize=np.shape(xi)
#print ip

#output matrix
op = np.empty((200,3))

file_location = r'training_and_testing.xlsx'
workbook1 = xlrd.open_workbook(file_location)
second_sheet = workbook.sheet_by_index(1)
for j in range (0,20):
  xo= [second_sheet.cell_value(j,i) for i in range (3)]
 # print xo
  op[j,:]=(xo)
#osize=np.shape(xo)
#print osize
def bp_function(ip=None, op=None, tr_cv=None, lr=None, iter=None, hneurons=None, msegoal=None):
    
    def bipsig(x=None):
        y = 1 / (1 + exp(-x))

        def bipsig1(x=None):
            y = bipsig(x) * (1 - bipsig(x))
    #Input arguments are: 
    #1)ip ->rows X cols: rows contain exemplars and cols contain features
    #2)op ->rows x cols: rows contain targets
    #3)tr_cv:keep 0.5
    #4)lr:learning rate (0.1<lr<0.5)
    #5)iterations:iterations
    #6)hneurons: Number of hidden layer neurons
    #7)msegoal:termination criterion for training on cv mse
    #
    #Outputs:
    #w1,w2: Weights of hidden layer and output layer
    #b1,b2:Biases of hidden layer and output layer
    #msetr:final training mse
    #msecv:final cv mse
    
   # [exemplars, R] = size(ip)
   # [exemplars, S2] = size(op)

    R = 200
    print R
    amp = np.zeros(R, 1)
    print amp
    off = np.zeros(R, 1)
    for i in mslice[1:R]:
        k = ip(mslice[:], mslice[i:i])
        amp(i, 1).lvalue = (2) / (max(k) - min(k))
        off(i, 1).lvalue = 1 - (amp(i, 1) * max(k))    # Normalizing input matrix
    end
    amp = amp.cT
    off = off.cT
    AMP = repmat(amp, xi, 1)
    OFF = repmat(off, xi, 1)
    nactips = zeros(xi, R)
    nip = AMP *elmul* ip + OFF
    nip = nip.cT
    op = op.cT

    tr_exemplars = uint32(tr_cv * exemplars)
    ptr = nip(mslice[:], mslice[1:tr_exemplars])
    ptr = ptr.cT
    ttr = op(mslice[:], mslice[1:tr_exemplars])
    ttr = ttr.cT
    pcv = nip(mslice[:], mslice[tr_exemplars:exemplars])
    pcv = pcv.cT
    tcv = op(mslice[:], mslice[tr_exemplars:exemplars])
    tcv = tcv.cT

    S1 = hneurons
    alpha = lr

    mf = 0.9

    v = rand(R, S1) - 0.5#R
    v1 = zeros(R, S1)
    b1 = rand(1, S1) - 0.5#S1,1
    b2 = rand(1, S2) - 0.5
    w = rand(S1, S2) - 0.5# S2,S1
    w1 = zeros(S1, S2)#S2,S1

    for epoch in mslice[1:iter]:
        e = 0
        ecv = 0; print ecv

        for I in mslice[1:tr_exemplars]:        #iter
            #feed forward
            for j in mslice[1:S1]:            # S1
                zin(j).lvalue = b1(j)
                for i in mslice[1:R]:                # tr_exemplars
                    zin(j).lvalue = zin(j) + ptr(I, i) *elmul* v(i, j)
                end
                z(j).lvalue = bipsig(zin(j))
            end
            for j in mslice[1:S1]:            # S1
                zincv(j).lvalue = b1(j)
                for i in mslice[1:R]:                # tr_exemplars
                    zincv(j).lvalue = zincv(j) + pcv(I, i) *elmul* v(i, j)
                end
                zcv(j).lvalue = bipsig(zincv(j))
            end
            for k in mslice[1:S2]:
                yin(k).lvalue = b2(k)
                for j in mslice[1:S1]:
                    yin(k).lvalue = yin(k) + z(j) *elmul* w(j, k)
                end
                y(k).lvalue = bipsig(yin(k))
                ty(I, k).lvalue = y(k)
            end
            for k in mslice[1:S2]:
                yincv(k).lvalue = b2(k)
                for j in mslice[1:S1]:
                    yincv(k).lvalue = yincv(k) + zcv(j) *elmul* w(j, k)
                end
                ycv(k).lvalue = bipsig(yincv(k))
                tycv(I, k).lvalue = ycv(k)
            end
            # Back propagation Error
            for k in mslice[1:S2]:
                delk(k).lvalue = (ttr(I, k) - y(k)) * bipsig1(yin(k))
            end
            for j in mslice[1:S1]:
                for k in mslice[1:S2]:
                    delw(j, k).lvalue = alpha * delk(k) * z(j) + mf * (w(j, k) - w1(j, k))
                    delinj(j).lvalue = delk(k) * w(j, k)
                end
            end
            delb2 = alpha * delk
            for j in mslice[1:S1]:
                delj(j).lvalue = delinj(j) * bipsig1(zin(j))
            end
            for j in mslice[1:S1]:
                for i in mslice[1:R]:
                    delv(i, j).lvalue = alpha * delj(j) * ptr(I, i) + mf * (v(i, j) - v1(i, j))
                end
            end
            delb1 = alpha * delj
            w1 = w
            v1 = v

            # weight update
            w = w + delw
            b2 = b2 + delb2
            v = v + delv
            b1 = b1 + delb1
            for k in mslice[1:k]:
                e = e + (ttr(I, k) - y(k)) ** 2
                earr = (mean(e))
            end
            for k in mslice[1:k]:
                ecv = ecv + (tcv(I, k) - ycv(k)) ** 2
                ecvarr = (mean(ecv) / 5)
            end

            #        
        end
        if e < msegoal:
            break
            # con=0;
        end
        e = mean(earr)
        ecv = mean(ecvarr / 2)
        error(epoch).lvalue = e
        errorcv(epoch).lvalue = ecv

        print(mstring('MSE(Training) : %f\\n'), error)

        print(mstring('Iteration : %f\\n'), epoch)
    end
    # 
    # plot(error,'--rs');
    # title('MSE v/s Iterations','Color','r');
    # xlabel('Iterations');
    # ylabel('MSE');
    # drawnow
    # hold on
    # plot(errorcv,'--b*');
    # drawnow
    # h = legend('TRG','CV',1);
    #disp(epoch);
    # disp(e);
    v()
    b1()
    w()
    b2()
