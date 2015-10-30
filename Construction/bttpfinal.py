# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:35:58 2014

@author: annakorba
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:04:23 2014

@author: jeremyt909
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:53:00 2014

@author: jeremyt909
"""


try:
    import numpy
except ImportError:
    print "numpy is not installed"
    
try:
    import csv
except ImportError:
    print "csv is not installed"

#try:
#    import matplotlib as plt
#except ImportError:
#    print "matplotlib is not installed"
     
try:
    import cPickle
except ImportError:
    print "cPickle is not installed"
    
try:
    import json
except ImportError:
    print "json is not installed"
    
    
from numpy import *
from auxFunctions import *
from sklearn.ensemble import RandomForestRegressor
#from sklearn.mixture import DPGMM
from sklearn.svm import SVR
from operator import add
#####

allInfo=numpy.array(list(csv.reader(open("../Data/bttp1.csv","rb"),delimiter=','))).astype('float')
allInfoHead={'page_id (0)','timestamp (1)', 'day (2)','minute (3)','fans (4)','type (5)','likes (6)','comments (7)','shares (8)','user_id (9)','country (10)'}
diffTime=numpy.array(list(csv.reader(open("../Data/diffTime.csv","rb"),delimiter=','))).astype('float')

    
###### change time minute 
N=allInfo.shape[0]
for i in range(0,N):
   I=(allInfo[i,9]==diffTime[:,0]).nonzero()[0]
   if I.shape[0]>0 :
       [allInfo[i,2], allInfo[i,3]]=hourAddMin(allInfo[i,2],allInfo[i,3],diffTime[I,1])
   
N=allInfo2.shape[0]      
for i in range(0,N):
   I=(allInfo2[i,9]==diffTime[:,0]).nonzero()[0]
   if I.shape[0]>0 :
      [allInfo2[i,2], allInfo2[i,3]]=hourAddMin(allInfo2[i,2],allInfo2[i,3],diffTime[I,1])
 
      
####### normalize by page 
uniPage= array(unique(allInfo[:,0]))
n=uniPage.size      
print 'pages count :',n 
allInfo[:,6]=multiply(allInfo[:,6],allInfo[:,4]**(-0.7))
allInfo[:,7]=multiply(allInfo[:,7],allInfo[:,4]**(-0.7))
allInfo[:,8]=multiply(allInfo[:,8],allInfo[:,4]**(-0.7))
#     
allInfo2[:,6]=multiply(allInfo2[:,6],allInfo2[:,4]**(-0.7))
allInfo2[:,7]=multiply(allInfo2[:,7],allInfo2[:,4]**(-0.7))
allInfo2[:,8]=multiply(allInfo2[:,8],allInfo2[:,4]**(-0.7))
print allInfo[5,6:9]
print allInfo2[5,6:9]

allInfo=allInfo[~np.isnan(allInfo).any(1)]
allInfo2=allInfo2[~np.isnan(allInfo2).any(1)]
##
##
###### suppression des doublons de publications
#
print 'suppression des doublons'

print allInfo.shape
allInfobis=copy(allInfo)
doublons=array([])
uniPage= array(unique(allInfo[:,1]))
n=uniPage.size  
for i in range(0,n):
    localpage= list((allInfo[:,1]==uniPage[i]).nonzero()[0])
    if len(localpage)>1:
        localpage.pop(0)
        doublons = append(doublons,localpage)
doublons = doublons.astype('int')    
allInfo=delete(allInfobis,doublons,axis=0)  
del allInfobis
print allInfo.shape
#
print allInfo2.shape
allInfobis2=copy(allInfo2)
doublons=array([])
uniPage= array(unique(allInfo2[:,1]))
n=uniPage.size  
for i in range(0,n):
    localpage= list((allInfo2[:,1]==uniPage[i]).nonzero()[0])
    if len(localpage)>1:
        localpage.pop(0)
        doublons = append(doublons,localpage)    
doublons = doublons.astype('int')    
allInfo2=delete(allInfobis2,doublons,axis=0)  
del allInfobis2
print allInfo2.shape


###### suppression des outliers 
#
print 'suppression des outliers'

allInfobis=copy(allInfo)
y=array(sum(allInfo[:,6:9],axis=1)) #likes comments shares
outliers=array([])
uniPage= array(unique(allInfo[:,0]))
n=uniPage.size  
for i in range(0,n):
    local= (allInfo[:,0]==uniPage[i]).nonzero()[0]
    thresold=mean(y[local])+2*std(y[local])
    outliers = append(outliers,local[(y[local]>thresold).nonzero()[0]])
outliers = outliers.astype('int')    
allInfo=delete(allInfobis,outliers,axis=0)  
del allInfobis 

#  
allInfobis2=copy(allInfo2)
y=array(sum(allInfo2[:,6:9],axis=1)) #likes comments shares
outliers=array([])
uniPage= array(unique(allInfo2[:,0]))
n=uniPage.size  
for i in range(0,n):
    local= (allInfo2[:,0]==uniPage[i]).nonzero()[0]
    thresold=mean(y[local])+2*std(y[local])
    outliers = append(outliers,local[(y[local]>thresold).nonzero()[0]])
outliers = outliers.astype('int')    
allInfo2=delete(allInfobis2,outliers,axis=0)  
del allInfobis2
print allInfo2.shape
#
###################### suppression des types non courants
#

print allInfo.shape
allInfobis=copy(allInfo)
types=array([])
n=len(allInfo)
for i in range(0,n):
    if allInfo[i,5]>2:
        if allInfo[i,5]!=4:
            if allInfo[i,5]!=9:
                    print allInfo[i,5]
                    types= append(types,i)
allInfo=delete(allInfobis,types,axis=0)  
del allInfobis 
print allInfo.shape

print allInfo2.shape
allInfobis2=copy(allInfo2)
types=array([])
n=len(allInfo2)
for i in range(0,n):
    if allInfo2[i,5]>2:
        if allInfo2[i,5]!=4:
            if allInfo2[i,5]!=9:
                    print allInfo2[i,5]
                    types = append(types,i)
types = types.astype('int')    
allInfo2=delete(allInfobis2,types,axis=0)  
del allInfobis2 
print allInfo2.shape


with open('../Models/allInfo.json', 'w') as outfile:  
       outfile.write(json.dumps(allInfo.tolist()))
with open('../Models/allInfo2.json', 'w') as outfile:  
       outfile.write(json.dumps(allInfo2.tolist()))   

############# separation train / test

#jsonfile=open('../Models/allInfo.json').read()
#allInfo=asarray(json.loads(jsonfile))
#jsonfile=open('../Models/allInfo2.json').read()
#allInfo2=asarray(json.loads(jsonfile))
#####
#
#N=allInfo2.shape[0]
#r=random.random(N);
#r1=r<0.75;
#r2=((r>=0.75).astype(int)==1);
#X= array(allInfo2[:,(0,2,3,4,5)])
#y=array(sum(allInfo2[:,6:9],axis=1))
#X_train= X[r1]
#y_train= y[r1]
#X_test= X[r2]
#y_test= y[r2]



#
#
#X_test=allInfo2[:,(0,2,3,4,5)]
#uniPageX=array(unique(X_test[:,0]))
#print uniPageX.size
#y_test=sum(allInfo2[:,6:9],axis=1)
#print X_test.shape
#X_test2=copy(X_test)
#y_test2=copy(y_test)
#uniPage2= array(unique(allInfo2[:,0]))
#a=uniPage2.size
#notcommon=array([])
#for i in range(0,a):
#    #localpage= (allInfo2[:,0]==uniPage[i]).nonzero()[0]
#    localpage= (allInfo[:,0]==uniPage2[i]).nonzero()[0]
#    if len(localpage)==0:
#        localpage2=(allInfo2[:,0]==uniPage2[i]).nonzero()[0]
#        notcommon=append(notcommon, localpage2)
#notcommon = notcommon.astype('int')    
#X_test=delete(X_test2,notcommon,axis=0)
#y_test=delete(y_test2,notcommon,axis=0)
#print X_test.shape,y_test.shape
#uniPageX=array(unique(X_test[:,0]))
#print uniPageX.size
#
#uniPage= array(unique(allInfo[:,0]))
#uniPage2= array(unique(allInfo2[:,0]))
#a=set(uniPage).intersection(uniPage2)
#    
#
############# randomForest prediction - entrainement sur le 1 er data set, test sur le 2eme.
#print "start training randomForest"

###### variables
#nbTree=50
#nbProc=2
#minSplit=10
#scoreMin=0.6
#nbTourMax=5
#minTree= 50
#maxTree= 51
#step=15
#
#score = 0 
#tour=0
#while (score <scoreMin and tour <nbTourMax):
#    clf = RandomForestRegressor(n_estimators=nbTree,oob_score=True,min_samples_split=minSplit,n_jobs=nbProc)
#    clf = clf.fit(X_train, y_train)
#    print 'score with ', minSplit, ' minSplit'
#    print clf.score(X_train, y_train)
#    score=clf.score(X_test,y_test)
#    tour=tour+1
#    print score
#
#with open('../Models/forest', 'wb') as f:
#    cPickle.dump(clf, f)
#
#print "random forest computed"

########################## construction de data2 ###################

############## engagement réel et engagement RF
#
#y=array(sum(allInfo2[:,6:9],axis=1)) 
#X=array(allInfo2[:,(0,2,3,4,5)])
with open('../Models/forest', 'rb') as f:
    clf = cPickle.load(f)
predAll=clf.predict(X_test)
#
################# construction de data2 et des dummy types
####
allI=allInfo2[r2]
#[ N , M ]= allInfo2.shape
[ N , M ]= allI.shape
print N,M
data2=zeros((N,M+3))
for i in range(0,N):
    #z=allInfo2[i,:]
    z=allI[i,:]
    for j in (0,1,2,3,4):
        data2[i,j]+=z[j]
    for j in (6,7,8,9,10):
        data2[i,j+3]+=z[j]
    #type=allInfo2[i,5]
    type=allI[i,5]
    if type==1:
        data2[i,5]+=1
    elif type==2:
        data2[i,6]+=1
    elif type==4:
        data2[i,7]+=1
    elif type==9:
        data2[i,8]+=1


############### normalisation des fans dans data2
#
yfan=[]
uniPage2= array(unique(data2[:,0]))
a=uniPage2.size
for i in range(0,a):
    localpage= (data2[:,0]==uniPage2[i]).nonzero()[0]
    yfan.append(mean(data2[localpage,4]))

data2[:,4]=(data2[:,4]-amin(yfan))/float(amax(yfan)-amin(yfan))


with open('../Models/data2bis.json', 'w') as outfile:  
       outfile.write(json.dumps(data2.tolist()))

with open('../Models/predAllbis.json', 'w') as outfile:  
       outfile.write(json.dumps(predAll.tolist()))
       
with open('../Models/ybis.json', 'w') as outfile:  
       outfile.write(json.dumps(y.tolist()))


##### page_id (0)','timestamp (1)', 'day (2)','minute (3)','fans (4)',
##### type lien (5), type photo (6), type video (7), type texte (8)
#####likes (9)','comments (10)','shares (11)','user_id (12)','country (13)'
#
################################ Construction des INPUT
#
jsonfile=open('../Models/data2bis.json').read()
data2=asarray(json.loads(jsonfile))
jsonfile=open('../Models/ybis.json').read()
y=asarray(json.loads(jsonfile))
jsonfile=open('../Models/predAllbis.json').read()
predAll=asarray(json.loads(jsonfile))

uniPage2= array(unique(data2[:,0]))
a=uniPage2.size
#
inputw=[]
targetw=[]
tsf=[]
tsfind=[]

for i in range(0,a):
    localpage= (data2[:,0]==uniPage2[i]).nonzero()[0]
    l=len(localpage)
    if l>5:
        for j in range(0,l-6):
            vectpage=[]
            for k in range(j+5,j-1,-1):
                fan=data2[localpage[-k],4]
                print 'fan'+str(fan)
                if k!= j:
                    ts=data2[localpage[-j],1]-data2[localpage[-k],1]
                    tsf.append(ts)
                    tsfind.append(localpage[-k])                  
                    vectpage.append(ts)
                    vectpage.append(fan)
                    vectpage.extend(list(data2[localpage[-k],(5,6,7,8)]))
                    vectpage.append(predAll[localpage[-k]])
                    vectpage.append(y[localpage[-k]])
                else:             
                    vectpage.append(fan)
                    vectpage.extend(list(data2[localpage[-k],(5,6,7,8)]))
                    vectpage.append(predAll[localpage[-k]])  
                    targ=y[localpage[-k]]
                    targetw.append(targ)
            inputw.append(vectpage)

####normalisation des timestamp
inputw2=asarray(inputw)
e=float(1)/float(amax(tsf)-amin(tsf))
for j in (0,8,16,24,32):
        inputw2[:,j]=e*(0.1*(amax(tsf)-inputw2[:,j])+0.9*(inputw2[:,j]-amin(tsf)))

with open('../Models/inputwbis.json', 'w') as outfile:  
       outfile.write(json.dumps(inputw2.tolist()))
                   
with open('../Models/targetwbis.json', 'w') as outfile:  
       outfile.write(json.dumps(targetw))


####### input: 0-7/8-15/16-23/24-30/31-38/39-45
#
################################# RESEAU DE NEURONES 


import sys
 
from pybrain.datasets            import SequentialDataSet
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.tools.neuralnets    import NNregression
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import LinearLayer,TanhLayer, SigmoidLayer, BiasUnit
from pybrain.structure.networks import Network

from pybrain.structure              import FullConnection
from pybrain.structure              import FeedForwardNetwork
#

#
jsonfile=open('../Models/inputwbis.json').read()
input=asarray(json.loads(jsonfile))
jsonfile=open('../Models/targetwbis.json').read()
target=asarray(json.loads(jsonfile))

#
N=len(target)
r=random.random(N);
r1=r<0.75;
r2=((r>=0.75).astype(int)==1);

inputtrain=input[r1];
targettrain= target[r1];
inputtest=input[r2];
targettest=target[r2];
#
trndata=SupervisedDataSet(46, 1)
tstdata=SupervisedDataSet(46, 1)
for i in range(0,inputtrain.shape[0]):
    trndata.addSample(inputtrain[i,:],targettrain[i])
for i in range(0,inputtest.shape[0]):  
    tstdata.addSample(inputtest[i,:],targettest[i])



########################## calcul MSE 

def rn(output):
    sumn=0
    sumr=0
    p=len(output)
    for i in range(0,p):
         sumr+=(inputtest[i,-1]-targettest[i])**2
         sumn+=(result[i]-targettest[i])**2  
    sumr=float(sumr)/float(2*p)
    sumn=float(sumn)/float(2*p)
    return sumr, sumn

######################## calcul du Rcarré
def rsquared(output):
    sst=0
    ssres=0
    p=len(output)
    for i in range(0,p):
        ssres+=(output[i]-targettest[i])**2
        sst+=(targettest[i]-mean(targettest))**2  
    s=1-(float(ssres)/float(sst))
    return s

######################## imprimer les poids
#
#def poids_connexions(n):
#    for mod in n.modules:
#        for conn in n.connections[mod]:
#            print conn
#            for cc in range(len(conn.params)):
#                print conn.whichBuffers(cc), conn.params[cc] 
#print poids_connexions(fnn)


############## TESTS SUCCESSIFS, TRACES


############### Tests à une couche cachée en faisant varier le nombre de hidden units (1cnbhu)
#
#outputs=[]
#trainvaliderror=[]
#testerror=[]
#r2=[]
#for i in (2,5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270,300, 350, 400): 
#    print i
#    fnn= buildNetwork( trndata.indim, i, trndata.outdim,bias= True, hiddenclass= SigmoidLayer, outclass= LinearLayer, recurrent=False)
#    fnn.randomize()
#    trainer = BackpropTrainer( fnn, dataset=trndata, learningrate=0.001, momentum=0.3, verbose=True, weightdecay=0.005)
#    [a,b]=trainer.trainUntilConvergence(trndata,maxEpochs=3)
#    trainvaliderror.append([a,b])
#    trainer.testOnData(tstdata)
#    result= fnn.activateOnDataset( tstdata )
#    outputs.append(result)
#    testerror.append(rn(result))
#    r2.append(rsquared(result))
#    print rsquared(result)

#for i in range(0,len(outputs)):
#    outputs[i]=[reduce(add,x) for x in outputs[i]]
#with open('../Models/outputs1cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(outputs))
#                   
#with open('../Models/trainvaliderror1cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(trainvaliderror))
#
#with open('../Models/testerror1cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(testerror))
#   
#with open('../Models/r21cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(r2))
#
#plt.clf()
#x=[2,5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200,250, 300, 350, 400]
#train=[0.004993 ,0.004716,0.00419 ,0.003681,0.003786,  0.003727,0.003213,0.003431,0.003407,0.003266,0.003094,0.003445, 0.003313,0.003296,0.003391,0.003308,0.003308,0.003144]
#valid=[0.00491,0.004253 ,0.004042,0.004799,0.003595,0.003262, 0.004088,0.003202, 0.002983,0.003138,0.003662,0.002776,0.002879, 0.00257, 0.003069,0.002663,0.002971,0.004284 ]
#fig, ax = plt.subplots()
#ax.plot(x,train, label= 'train MSE' )
#ax.plot(x,valid, label= 'validation MSE' )
#ax.set_xlabel('Nombre de neurones')
#ax.set_ylabel('MSE')
##plt.ylim(0,1)
#xticks((10, 30, 50, 100, 150, 200,250,300, 350, 400))
#ax.legend(loc=1);
##yticks(arange(0,1,0.1))
#fig.savefig("trainvalid1cnbhu.png")

################ Tests à deux couches cachées en faisant varier le nombre de hidden units (2cnbhu)
#outputs=[]
#trainvaliderror=[]
#testerror=[]
#r2=[]
#for i in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120, 130, 140, 150, 160, 170, 180, 190, 200,210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 350):
#    outputsbis=[]
#    trainvaliderrorbis=[]
#    testerrorbis=[]
#    r2bis=[]
#    for j in(10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110, 120, 130, 140, 150, 160, 170, 180, 190, 200,210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 350):
#        print i,j
#        fnn= buildNetwork( trndata.indim, i, j, trndata.outdim,bias= True, hiddenclass= SigmoidLayer, outclass= LinearLayer, recurrent=False)
#        fnn.randomize()
#        trainer = BackpropTrainer( fnn, dataset=trndata, learningrate=0.001, momentum=0.3, verbose=True, weightdecay=0.005)
#        [a,b]=trainer.trainUntilConvergence(trndata,maxEpochs=3)
#        trainvaliderrorbis.append([a,b])
#        trainer.testOnData(tstdata)
#        result= fnn.activateOnDataset( tstdata )
#        outputsbis.append(result)
#        testerrorbis.append(rn(result))
#        r2bis.append(rsquared(result))
#        print rsquared(result)
#    outputs.append(outputsbis)
#    trainvaliderror.append(trainvaliderrorbis)
#    testerror.append(testerrorbis)
#    r2.append(r2bis)



#with open('../Models/outputs2cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(outputs))
#   
#with open('../Models/trainerror2cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(trainerror))                
#with open('../Models/validerror2cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(validerror))

#with open('../Models/testerror2cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(testerror))
#
#with open('../Models/r22cnbhu.json', 'w') as outfile:  
#       outfile.write(json.dumps(r2))
#      
#      
#
#matrice = array(r22cnbhu)
#img = plt.matshow(matrice)
#plt.colorbar(img)
#labelPositions = [0,1,2,3,4,5]
#newLabels = ['10', '20', '50', '100','150', '200']
#plt.xticks(labelPositions,newLabels)
#plt.yticks(labelPositions,newLabels)
#plt.xlabel('1ere couche')
#plt.ylabel('2eme couche')
#plt.show
#plt.savefig("validerror2cnbhu.png")


############## Tests à 1 couche cachée en faisant varier les paramètres de réglage (lmw)

#outputs=[]
#outputsbis=[]
#trainvaliderror=[]
#testerror=[]
#r2=[]
#for m in (0.1,0.2,0.3,0.4,0.5):
#    outputsbis=[]
#    trainvaliderrorbis=[]
#    testerrorbis=[]
#    r2bis=[]
#    for w in (0.0005,0.001,0.005,0.01, 0.05):
#        print m,w
#        fnn= buildNetwork( trndata.indim,300, trndata.outdim,bias= True, hiddenclass= SigmoidLayer, outclass= LinearLayer, recurrent=False)
#        fnn.randomize()
#        trainer = BackpropTrainer( fnn, dataset=trndata, learningrate=0.001, momentum=m, verbose=True, weightdecay=w)
#        [a,b]=trainer.trainUntilConvergence(trndata,maxEpochs=3)
#        trainvaliderrorbis.append([a,b])
#        trainer.testOnData(tstdata)
#        result= fnn.activateOnDataset( tstdata )
#        outputsbis.append(result)
#        testerrorbis.append(rn(result))
#        r2bis.append(rsquared(result))
#    outputs.append(outputsbis)
#    trainvaliderror.append(trainvaliderrorbis)
#    testerror.append(testerrorbis)
#    r2.append(r2bis)
#
#    
#with open('../Models/outputs1clmw.json', 'w') as outfile:  
#       outfile.write(json.dumps(outputs))
#                   
#with open('../Models/trainvaliderror1clmw.json', 'w') as outfile:  
#       outfile.write(json.dumps(trainvaliderror))
#
#with open('../Models/testerror2clmw.json', 'w') as outfile:  
#       outfile.write(json.dumps(testerror))
#
#with open('../Models/r22clmw.json', 'w') as outfile:  
#       outfile.write(json.dumps(r2))


################# graphiques
###
fnn= buildNetwork( trndata.indim, 300, trndata.outdim,bias= True, outputbias= True,hiddenclass= SigmoidLayer, outclass=LinearLayer, recurrent=False)
fnn.randomize()
trainer = BackpropTrainer( fnn, dataset=trndata, learningrate=0.001, momentum=0.3, verbose=True, weightdecay=0.005)   
[a,b]=trainer.trainUntilConvergence(maxEpochs=3)
trainer.testOnData(tstdata)
result= fnn.activateOnDataset( tstdata )
print rn(result)
print rsquared(result)
#
#plt.clf()
#y1=targettest
#y2=result
##y2bis=list(itertools.chain(*y2))
#y3=inputtest[:,-1]
##idth = 0.04 
#plt.hist((y1,y2, y3), bins= 40, cumulative=False,label= (' target ', 'output', 'rf'))
#
##plt.hist(y1, cumulative=False)
#plt.title("Frequence des engagements")
#plt.xlabel("Engagement")
#plt.ylabel("Frequence")
#plt.legend(loc=1);
#xticks(arange(-0.2,0.8,0.1))
#plt.xlim(-0.2,1)
#plt.show()
#plt.savefig("300.png")
#
#
#x=range(0,51)
#fig, ax = plt.subplots()
#train=[0.113348 , 0.01125  , 0.00471  , 0.003691 , 0.003571 , 0.0036   , 0.003825 , 0.004021 , 0.004321 , 0.004666 , 0.004915 , 0.005146 , 0.005303 , 0.00541  , 0.005496 , 0.005563 , 0.005614 , 0.005637 , 0.00567  , 0.005693 , 0.00573  , 0.005743 , 0.005748 , 0.005768 , 0.005781 , 0.005786 , 0.005787 , 0.005805 , 0.005808 , 0.005789 , 0.005789 , 0.005797 , 0.005802 , 0.005805 , 0.005803 , 0.005809 , 0.0058   , 0.005803 , 0.005805 , 0.005807 , 0.005808 , 0.005809 , 0.005814 , 0.005813 , 0.005809 , 0.005821 , 0.005803 , 0.005814 , 0.005809 , 0.005799 , 0.00581  ]
#valid=[0.01868  , 0.005592 , 0.003463 , 0.00319  , 0.00393  , 0.003414 , 0.003402 , 0.003745 , 0.004298 , 0.004096 , 0.004275 , 0.004955 , 0.004497 , 0.004754 , 0.004625 , 0.004717 , 0.004775 , 0.004742 , 0.004909 , 0.004786 , 0.004856 , 0.005004 , 0.004879 , 0.004819 , 0.004868 , 0.005216 , 0.004847 , 0.005062 , 0.004822 , 0.004895 , 0.005024 , 0.005219 , 0.005258 , 0.004948 , 0.005034 , 0.004835 , 0.005014 , 0.004999 , 0.005121 , 0.004854 , 0.004953 , 0.004972 , 0.004964 , 0.004905 , 0.004834 , 0.004834 , 0.005046 , 0.005116 , 0.004979 , 0.004837 , 0.004869] 
#ax.plot(x,train, label= 'train MSE' )
#ax.plot(x,valid, label= 'validation MSE' )
#ax.set_xlabel('Nombre de neurones')
#ax.set_ylabel('MSE')
##plt.ylim(0,1)
#xticks((0,3,10,20,30,40,50))
#ax.legend(loc=1);
##yticks(arange(0,1,0.1))
#fig.savefig("trainvalidepochs.png")

########### ############## ############# ############## Tests sur les inputs

#import matplotlib.pyplot as pyplot
#pyplot.scatter(data2[:,4], y)
#pyplot.scatter(predAll, y)


##### ACp
#pyplot.scatter(y, X[:,4])

