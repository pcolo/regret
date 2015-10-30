# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 19:06:24 2014

@author: Hélène
"""
import matplotlib.image as mpimg
import scipy.stats as st
import scipy.linalg as la
import pandas

def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # source: http://www.johndcook.com/python_longitude_latitude.html 
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians         
    # Compute spherical distance from spherical coordinates.
    cos = math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + math.cos(phi1)*math.cos(phi2)
    arc = math.acos( cos )
    return arc
 
# stations_within_r: returns the IDs of all stations within a distance r (km) of the center of
# the region defined by the $regionID
def stations_within_r(regionID, r):
    # read data from files
    regions = pandas.read_csv("_RegionsData.csv")
    stations = pandas.read_csv("_StationsData.csv")
    # extract latitude and longitude of the center of the region
    Rlatitude = float(regions[regions['ABR'] == regionID]['LATITUDE'])
    Rlongitude = float(regions[regions['ABR'] == regionID]['LONGITUDE']) 
    # compute the distance to every station
    stations['DISTANCE'] = 0
    for ix in range(len(stations.index)-1):
        stations.loc[ix, 'DISTANCE'] = 6373*distance_on_unit_sphere(Rlatitude, Rlongitude,float(stations.loc[ix, 'LATITUDE']), float(stations.loc[ix, 'LONGITUDE']))
    # return stations IDs
    return stations[stations['DISTANCE']<r]
    #return stations.loc[stations['DISTANCE'] <= r, '\ufeffID'].tolist()

#img=mpimg.imread('German_postcode_information.png')
#imgplot = plt.imshow(img)
#larg=902-1
#haut=1216-1
#minx=5.53
#maxx=15.02
#minh=47.24
#maxh=55.03
# stations positioning in the ZIP codes plane
#ref,xcoord,ycoord=np.genfromtxt('C:\Users\Ln\Documents\Machine Learning\Stations_1990-2012.csv',delimiter=',',skip_header=1,usecols = (0,4,5),unpack=True);
#plot((ycoord-minx)*larg/(maxx-minx),haut-(xcoord-minh)*haut/(maxh-minh),'o')
# identification of the stations through their ID
#for i in range(len(ref)):
#    text((ycoord[i]-minx)*larg/(maxx-minx),haut-(xcoord[i]-minh)*haut/(maxh-minh),ref[i],fontsize=7, bbox=dict(facecolor='cyan',alpha=0.5))

#PICP
def PICP(y,L,U):
    c=(y>=L) & (y<=U)
    size_c=c.size
    return 1.0*np.sum(c)/size_c
    
# Score
def Score(y,L,U,alpha):
    delta=U-L
    Sc=np.zeros(L.shape)
    for i in range(y.shape[0]): # parcours ligne
        for j in range(y.shape[1]): # parcours colonne
            if y[i,j]<L[i,j]:
                Sc[i,j]=-2.0*alpha*delta[i,j]-4.0*(L[i,j]-y[i,j])
            elif y[i,j]<=U[i,j]:
                Sc[i,j]=-2.0*alpha*delta[i,j] 
            else:
                Sc[i,j]=-2.0*alpha*delta[i,j]-4.0*(y[i,j]-U[i,j])
    size_Sc=Sc.size
    return 1.0*np.sum(Sc)/size_Sc

def ScoreV2(L,U):
    return np.sum(U-L)/L.size
   
# hidden-layer output matrix
def H(w,b,x):
    ###### multiquadratics function ######
    #x2=np.sum(x**2,axis=1)
    #x2=x2.reshape((x2.shape[0],1))    
    #w2=np.sum(w**2,axis=0)
    #w2=w2.reshape((1,w2.shape[0])) 
    #res=x2+w2-2*np.dot(x,w)    
    #return np.sqrt(res+b**2)
    ######################################
    ###### sigmoid function ######    
    res=(np.dot(x,w)+b)    
    return 1.0/(1.0+np.exp(-res))
    ##############################
    ###### oldies
    #poids=safe_sparse_dot(x, w) + b
    #return 1/(1+exp(-poids))
    #X=np.tile(x.reshape((x.shape[0],1)),w.shape[1])
    #return 1.0/(1.0+np.exp(-(w*X+b)))    

# computation of the ELM input/output weigths
def neurons(x,y,nb_neurons):
    n=x.shape[1]
    # random generation of the neurons parameters
    w=st.norm.rvs(size=(n, nb_neurons)) 
    b=st.norm.rvs(size=(1,nb_neurons))
    h=H(w,b,x) # activation matrix computation
    beta_chapeau=dot(la.pinv2(h),y) # Penrose-Moore inversion
    return w,b,beta_chapeau

# estimation of the target
def predict(x,w,b,beta_chapeau):
    h=H(w,b,x)
    return dot(h,beta_chapeau)

# pairs bootstrap
def bootboot(x,y):
    N=x.shape[0] # nombre de lignes
    B=np.random.randint(N,size=N)
    return x[B],y[B]

# ELM applied to the l-th Bootboot
#def ELM_bootboot(x,y,nb_neurons):
    #reg=GenELMRegressor(hidden_layer=RBFRandomLayer(n_hidden=nb_neurons, rbf_width=0.8))
    #reg=GenELMRegressor(hidden_layer=MLPRandomLayer(n_hidden=nb_neurons, activation_func='tanh'))
    #reg.fit(x, y)
    #reg=neurons(x,y,nb_neurons)    
    #return reg # classifier

# BM bootboot replicates
def Bootboot_replicate(x_train,y_train,nb_neurons,BM,x_test):
    l_boot=np.zeros((BM,x_test.shape[0],y_test.shape[1]))
    l_boot2=np.zeros((BM,x_train.shape[0],y_train.shape[1]))
    for i in range(BM):
        xB,yB=bootboot(x_train,y_train)
        w,b,beta_chapeau=neurons(xB,yB,nb_neurons)
        yl_chapeau=predict(x_test,w,b,beta_chapeau)
        l_boot[i,:,:]=yl_chapeau
        yl_chapeau2=predict(x_train,w,b,beta_chapeau)
        l_boot2[i,:,:]=yl_chapeau2
    return l_boot,l_boot2

# average output of the ensemble of BM ELMs
def Moymoy(BM,l_boot): 
    y_chapeau=np.zeros((l_boot.shape[1],l_boot.shape[2]))
    for i in range(BM):
        yl_chapeau=l_boot[i,:,:] 
        #plot(yl_chapeau-y)
        #if i==0:
        #    y_chapeau=yl_chapeau
        #else:
        y_chapeau+=yl_chapeau
    return y_chapeau*(1.0/BM)

# variance in the outputs of the BM ELMs
def Sig_y_chapeau(BM,l_boot,y_chapeau):
    #plot(y_chapeau)    
    for i in range(BM):
        yl_chapeau=l_boot[i,:,:]
        if i==0:
            sigma=(yl_chapeau-y_chapeau)**2
        else:
            sigma+=(yl_chapeau-y_chapeau)**2
    return sigma/(BM-1.0)

# computation of the upper and lower bounds of the PI
def Bobo(y_chapeau,sigma,critical):
    sigma=np.abs(sigma)
    return y_chapeau-critical*np.sqrt(sigma),y_chapeau+critical*np.sqrt(sigma)
    #return y_chapeau,y_chapeau

# root mean square error
def RMSE(y_chapeau,y):
    N_test=y.shape[0]
    return np.sqrt(1.0/N_test*np.sum((y-y_chapeau)**2))
    
# mean absolute error    
def MAE(y_chapeau,y):
    N_test=y.shape[0]
    return 1.0/N_test*np.sum(np.abs(y-y_chapeau))

# computation of the RMSE and MAE as functions of the number of hidden neurons
def Optim_neurons(nb_neurons,x,y):
    packs = np.random.permutation(range(x.shape[0])).reshape((-1,x.shape[0]/10)) #decoupage du set en 10 morceaux
    RMS = 0.0
    MA = 0.0    
    for i in range(packs.shape[0]):
        test_set = packs[i]
        train_set = packs[np.arange(packs.shape[0])!=i].ravel()
        w,b,beta_chapeau=neurons(x[train_set],y[train_set],nb_neurons)
        y_chapeau=predict(x[test_set],w,b,beta_chapeau)
        RMS+=RMSE(y_chapeau,y[test_set])
        MA+=MAE(y_chapeau,y[test_set])
    return RMS/packs.shape[0],MA/packs.shape[0]

# cross-validation to optimize the number of hidden neurons
def Cross_validation(x,y,liste_nb_neurons):
    cv_RMSE=np.zeros((size(liste_nb_neurons),1))
    cv_MAE=np.zeros((size(liste_nb_neurons),1))
    for nb_neurons in range(size(liste_nb_neurons)):
        cv_RMSE[nb_neurons],cv_MAE[nb_neurons]=Optim_neurons(liste_nb_neurons[nb_neurons],x,y)
    return cv_RMSE,cv_MAE    

# draw a graph for cross_validation
def cross_trace(x,y,liste_nb_neurons):
    cv_RMSE,cv_MAE=Cross_validation(x,y,liste_nb_neurons)
    plot(liste_nb_neurons,cv_RMSE,label='RMSE test')
    plot(liste_nb_neurons,cv_MAE,label='MAE test')
    xlabel('Number of ELM hidden neurons')
    ylabel('Validation test')
    legend()

#data extraction
def dat_extract(station,fichier,bystation=True):
    allstations_data = pandas.read_csv(fichier)    
    if bystation:
        stations = stations_within_r(station,100).iloc[:,0].tolist()        
        stations = list( set(allstations_data.columns.tolist()) & set(stations) )
        return allstations_data.loc[:,stations].values
    else:
        return allstations_data.loc[:,station].values
    
def add_data(station,fichier,x_train,x_test,bystation=True):
    data = dat_extract(station,fichier,bystation)
    if bystation:
        train = data[np.arange(350*24),:].reshape((-1,data.shape[1]*24))
        test = data[100*24+np.arange(7*24),:].reshape((-1,data.shape[1]*24))    
    else:
        train = data[np.arange(350*48)].reshape((-1,48))
        test = data[100*48+np.arange(7*48)].reshape((-1,48))
    x_train = np.concatenate((x_train,train),axis=1)    
    x_test = np.concatenate((x_test,test),axis=1)
    return x_train,x_test
    
# test avec solar2013
def PredSolar(station):
    dat=pandas.read_csv('solar2013.csv')
    donnees=np.array(dat[station])
    #train_period=np.arange(10*48)
    #test_period=10*48+np.arange(5*48)
    #x_train=donnees[train_period].astype('float').reshape((-1,1)) 
    #y_train=donnees[train_period+1].astype('float').reshape((-1,1))
    #x_test=donnees[test_period].astype('float').reshape((-1,1))
    #y_test=donnees[test_period+1].astype('float').reshape((-1,1))
    # prediction day D-1 to day D
    train_period_D=np.arange(350*48)
    test_period_D=100*48+np.arange(7*48)    
    x_train=donnees[train_period_D].astype('float').reshape((-1,48))    
    y_train=donnees[train_period_D+48].astype('float').reshape((-1,48))
    x_test=donnees[test_period_D].astype('float').reshape((-1,48))
    y_test=donnees[test_period_D+48].astype('float').reshape((-1,48))
    #x_train,x_test=add_data(station,'cloud2013.csv',x_train,x_test)
    #x_train,x_test=add_data(station,'temperature.csv',x_train,x_test)    
    #x_train,x_test=add_data(station,'WindSpeed2013.csv',x_train,x_test)
    #x_train,x_test=add_data(station,'WindDirection2013.csv',x_train,x_test)
    x_train,x_test=add_data(station,'Precipitation2013.csv',x_train,x_test)    
    #x_train,x_test=add_data(station,'Pressure2013.csv',x_train,x_test)
    #x_train,x_test=add_data(station,'Azimuth2013.csv',x_train,x_test,False)    
    return x_train, y_train, x_test, y_test

# algorithme du exponentially weighted average forecaster
# pred est l'ensemble des predictions de nos experts
# la premiere dimension donne le numero de l'expert
# la seconde dimension concerne le numero du sample
# la derniere dimension permet de parcourir toutes les features de sortie
def bandit(pred,y_train):
    eta = np.sqrt(2*np.log(pred.shape[0])/pred.shape[1])    
    w=np.ones((1,pred.shape[0])).astype('float')/pred.shape[0]
    for i in range(pred.shape[1]):
        loss=np.sum((y_train[i,:]-pred[:,i,:])**2,axis=1)
        w = w*np.exp(-eta*loss)
        w = w/np.sum(w)
    return w

#retourne les bornes infs et sups pour un intervalle de confiance de 1-alpha
def bounds(w,predL,predU,alpha):
    w=w.ravel()
    nb=min(w.size-np.sum((-np.cumsum(np.sort(-w)))>(1-alpha))+1,w.size)
    print nb,'/',w.size,'experts selectionnes'    
    indices = np.argsort(-w)[np.arange(nb)]
    predicL = predL[indices].reshape((indices.size,-1))
    predicU = predU[indices].reshape((indices.size,-1))
    return np.max(predicL,axis=0).reshape((predL.shape[1],predL.shape[2])),np.min(predicU,axis=0).reshape((predL.shape[1],predL.shape[2]))

nb_neurons=50 
BM=200
BN=100
#critical=1.6449 # confidence level = 95%
critical=1.2816 # confidence level = 90%
alpha=0.1
station='BY'

def data_for_experts(station,fichier):
    data = dat_extract(station,'solar2013.csv',False)    
    base_train = data[np.arange(350*48)].reshape((-1,48))
    base_test = data[100*48+np.arange(7*48)].reshape((-1,48)) 
    data = dat_extract(station,fichier)
    base_tr=np.zeros((data.shape[1],base_train.shape[0],base_train.shape[1]))+base_train
    base_te=np.zeros((data.shape[1],base_test.shape[0],base_test.shape[1]))+base_test    
    train = np.concatenate((base_tr,data[np.arange(350*24),:].reshape((data.shape[1],-1,24))),axis=2)
    test = np.concatenate((base_te,data[100*24+np.arange(7*24),:].reshape((data.shape[1],-1,24))),axis=2)    
    return train,test

def algo_modified(y_train,y_test,station,fichier):
    x_train,x_test=data_for_experts(station,fichier)
    L = np.zeros((x_train.shape[0],y_test.shape[0],y_test.shape[1]))
    U = np.zeros((x_train.shape[0],y_test.shape[0],y_test.shape[1]))
    m2 = np.zeros((x_train.shape[0],y_train.shape[0],y_train.shape[1]))
    for i in range(x_train.shape[0]):
        l,l2=Bootboot_replicate(x_train[i],y_train,nb_neurons,BM,x_test[i])
        m=Moymoy(BM,l)
        m2[i]=Moymoy(BM,l2)
        s=Sig_y_chapeau(BM,l,m)
        l_noise,l2=Bootboot_replicate(x_train[i],(m2[i]-y_train)**2,nb_neurons,BN,x_test[i])
        m_noise=Moymoy(BN,l_noise)
        s_noise=Sig_y_chapeau(BN,l_noise,m_noise)
        var_tot=m_noise+s_noise+s
        L[i],U[i] = Bobo(m,var_tot,critical)
    #data = dat_extract(station,'solar2013.csv',False)    
    #train = data[np.arange(350*48)].reshape((-1,48))
    #test = data[100*48+np.arange(7*48)].reshape((-1,48))    
    #l,l2=Bootboot_replicate(train,y_train,nb_neurons,BM,test)
    #m[m.shape[0]-1]=Moymoy(BM,l)
    #m2[m.shape[0]-1]=Moymoy(BM,l2)
    w = bandit(m2,y_train)
    return bounds(w,L,U,0.1)

def algo(x_train,y_train,x_test):
    l,l2=Bootboot_replicate(x_train,y_train,nb_neurons,BM,x_test)
    m=Moymoy(BM,l)
    m2=Moymoy(BM,l2)
    s=Sig_y_chapeau(BM,l,m)
    #cross_trace(x_train,(m2-y_train)**2)
    l_noise,l2=Bootboot_replicate(x_train,(m2-y_train)**2,nb_neurons,BN,x_test)
    m_noise=Moymoy(BN,l_noise)
    s_noise=Sig_y_chapeau(BN,l_noise,m_noise)
    var_tot=m_noise+s_noise+s
    return Bobo(m,var_tot,critical)

def algo_trace(x_train,y_train,x_test,y_test):
    #L,U=algo(x_train,y_train,x_test)
    L,U=algo_modified(y_train,y_test,station,'Precipitation2013.csv')
    print PICP(y_test,L,U)
    print Score(y_test,L,U,alpha)
    print ScoreV2(L,U)
    plot(L.ravel(),label='Lower Bound of PI')
    plot(U.ravel(),label='Upper Bound of PI')
    plot(y_test.ravel(),'o',label='Measured Value')
    xlabel('Time (1/2 hour)')
    ylabel('Solar Production')
    legend(loc='upper left')

x_train,y_train,x_test,y_test=PredSolar(station)
algo_trace(x_train,y_train,x_test,y_test)
#cross_trace(x_train,y_train,np.arange(1,100,1))
#algo(x_train,y_train,x_test)

### vieilles illustrations
#N=2*3*30
#x=100*rand(N)
#y=cos(x)
#y=rand(1)*log(x)**2
#x=x.reshape((x.shape[0],1)) 
#y=y.reshape((y.shape[0],1)) 
#algo_trace(x,y,x,y)
#w,b,beta_chapeau=neurons(x,y,nb_neurons)
#print predict(x,w,b,beta_chapeau), y