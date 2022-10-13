from numpy import loadtxt
import numpy as np
import random
from numpy import linalg as LA
import time
import pickle

from functions import *
from params import *

np.disp('Running bias memory allocation on 10 seeds and different current2set')

# 13/06 works 70% for competition
N=60
NCS=30
taur=20
tauw=700
zetaLA=0
rLA0 = np.zeros(N)
WLALA0 = np.zeros(N*N)
Wlow = .2
WCS0 = np.zeros((N,NCS)) + Wlow
WCS0[:int(N/4),:int(NCS/3)] = .3
WCS0[int(N/4):int(N/2),int(NCS/3):2*int(NCS/3)] = .3
WCS0[int(N/2):3*int(N/4),2*int(NCS/3):] = .3
I0 = 6
I1 = .9
threshold = 5
threshold2 = threshold
E = 3.5
tauf =  taur
taus =  12 * 3600 * 1000
CS = 4
US = 1
extra = 4
Nextra = 7
Ncut = 7
current1set = .2
current2set = .2
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000 

# protocol
training = []
for i in range(0,Nstim):
    training.append(100 + i*delay_between_event)
    training.append(100 + i*delay_between_event + delay_stimulation)
training = np.array(training)
recall = np.array([100,100+delay_stimulation])
index=[range(0,N),
               range(N,N + N*N),
               range(N + N*N,N + N*N +N)]
               
               
               
# main function
def f(yt,t,tag,dt):
    global theta
    rLA = yt[index[0]][np.newaxis].T
    rLA = rLA*(rLA>1e-5)

    WLALA = yt[index[1]].reshape((N,N))
    exc =   yt[index[2]][np.newaxis].T
    
    INPUT_rCS = WCS0.dot(rCS(t).T)[np.newaxis].T
    

    rinhib = I0 + np.sum(rLA)*I1
        
    drLAdt = (-rLA + np.maximum(0,WLALA.dot(rLA) + INPUT_rCS - rinhib + exc + block(t)[np.newaxis].T) )/taur
        
    D = 15000
    if t>D:
        delta = D
    else:
        delta = t
        
    theta = theta*(1-dt/D)+rLA*dt/D # approx : r(t-D) = theta

    dWLALAdt = rUS(t) * np.tanh((rLA).dot((rLA-theta).T)) / tauw
    dWLALAdt = np.multiply(dWLALAdt,np.logical_not(np.logical_or(np.logical_and(WLALA>=1,dWLALAdt>0),np.logical_and(WLALA<=0,dWLALAdt<0))))

                          
    # excitability
    dexcdt=np.zeros(N)
    for i in range(N):
        
        if t > plus and t < plus + max(training)-200:
            if i < Nextra: 
                dexcdt[i] = (extra + exc0[i] - exc[i]) / tauf  
                
        if t > plus + max(training)-200 and t < plus + max(training):
            if i < Nextra: 
                dexcdt[i] = (exc0[i]- exc[i]) / tauf
                
        if rLA[i] > threshold:
            if tag[i] == 0:
                tag[i] = t
                
        if t > tag[i] + max(training)-100 and t < tag[i] + max(training) :
            if tag[i] !=0:
                dexcdt[i] = (E + exc0[i] - exc[i]) / tauf
                
        if t > tag[i] + max(training):
            if tag[i] !=0:
                dexcdt[i] = (exc0[i]- exc[i]) / taus
                


    dydt = np.concatenate((drLAdt.flatten(),
                           dWLALAdt.reshape((N*N)).flatten(),
                           dexcdt.flatten()))
    
    return dydt




Nrep = 10

L_current2 = [.1,.2,.3,.4,.5]
freezing1 =    np.zeros((Nrep,4*len(L_current2)))
freezing1bis = np.zeros((Nrep,4*len(L_current2)))

current1 = 0
extraset = extra


for seed in range(0,Nrep):
	np.disp('Seed = '+str(seed+1))
	np.random.seed(seed+1)
	exc0 = np.random.normal(0,.5,N) 
	exc0 = np.sqrt(exc0*exc0)
	cc = 0
	for current2set in L_current2:
		for extra in [0,extraset]:
			for current2 in [0,current2set]:		
				# Parameters
				seqrCS1 = np.concatenate(( training, od + recall ))+plus
				seqrCS2 = []
				seqrCS3 =  []
				seqrUS = training+plus
				T_list = [0,plus,max(training)+max(training)+plus, od+plus, od +300+plus]
				dt_list = [.5,.5,20000,.5]
				rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)
	
				def block(t):
					L1 = range(Ncut) # inhibit only a subset of neurons
					#L1 = range(5) # inhibit only a subset of neurons
					#L2 = range(10) # inhibit only a subset of neurons
					L2 = range(Ncut) # inhibit only a subset of neurons
	
					Lneurons1 = np.zeros(N)
					Lneurons2 = np.zeros(N)
	
					for i in L1:
						Lneurons1[i] = 1
					for i in L2:
						Lneurons2[i] = 1
					if (t > plus  and t < plus  + max(training)):
						return - Lneurons1 * current1
					if (t > plus + od):
						return - Lneurons2 * current2
					else:
						return np.zeros(N)
	
				# Run
				y = run(rCS,rUS,time_steps,f)



				# Compute freezing

				recCS1 = y[index[0][:],np.where(time_steps == plus + od + 100)[0][0]:np.where(time_steps == plus + od + 200)[0][0]]

				freezing1[seed,cc] = np.sum(np.mean(np.maximum(0,recCS1), axis = 1))
				freezing1bis[seed,cc] = np.sum(np.mean(np.maximum(0,recCS1 - threshold), axis = 1))
				cc+=1
			
		np.disp([seed,current2set,freezing1[seed,:],freezing1bis[seed,:]])		



with open('freezing1_biasallocationLOOP_14_06.pkl', 'wb') as f:
	pickle.dump([freezing1], f)
with open('freezing1bis_biasallocationLOOP_14_06.pkl', 'wb') as f:
	pickle.dump([freezing1bis], f)
