from numpy import loadtxt
import numpy as np
import random
from numpy import linalg as LA
import time
import pickle

def run(rCS,rUS, time_steps,f):
    global theta
    theta = 0
    global step
    step = 0
    nstep = len(time_steps)
    thetaL=np.zeros((N,1))
    tag = np.zeros(N)
    y0 = np.concatenate((rLA0,WLALA0,exc0))
    y = np.zeros((len(y0),nstep))
    y[:,0] = y0
    for step in range(nstep-1):
        dt = time_steps[step+1] - time_steps[step]
        y[:,step+1] = (y[:,step] + dt * f(y[:,step],time_steps[step],tag,dt)[np.newaxis]);
    return y

def protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list):

    time_steps =[]
    for i in range(len(T_list)-1):  
        time_steps = np.concatenate(( time_steps , np.arange(T_list[i],T_list[i+1],dt_list[i]) ))

    def rCS(t):
        L1 = 0
        pol = 1
        for step in seqrCS1:
            L1 += np.tanh(t-step)*pol
            pol *= -1

        L2 = 0
        pol = 1
        for step in seqrCS2:
            L2 += np.tanh(t-step)*pol
            pol *= -1

        L3 = 0
        pol = 1
        for step in seqrCS3:
            L3 += np.tanh(t-step)*pol
            pol *= -1

            #US = 10
            

    
        return np.concatenate((np.ones(int(NCS/3)) * L1/2 , np.zeros(int(NCS/3))               , np.zeros(int(NCS/3))))*CS  + \
               np.concatenate((np.zeros(int(NCS/3))       , np.ones(int(NCS/3)) * L2/2         , np.zeros(int(NCS/3))))*CS  + \
               np.concatenate((np.zeros(int(NCS/3))       , np.zeros(int(NCS/3))               , np.ones(int(NCS/3)) * L3/2))*CS
    
    def rUS(t):
        L = 0
        pol = 1
        for step in seqrUS:
            L += np.tanh(t-step)*pol
            pol *= -1
        return US * np.ones(N)*L/2 + 1
    
    return(rCS,rUS,time_steps)
    




# np.random.seed(5)

# 27/06
N=60
NCS=30
taur=20
tauw=750
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
threshold = 6
threshold2 = threshold
E = 3.5
tauf =  taur
taus =  24 * 3600 * 1000
CS = 4
US = 1
extra = 4
Nextra = 7
Ncut = 5
current1set = 2
current2set = 2
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000

global theta
theta = 0
Nstim = 20
delay_between_event = 150
delay_stimulation = 40

# protocol
training = []
for i in range(0,Nstim):
    training.append(100 + i*delay_between_event)
    training.append(100 + i*delay_between_event + delay_stimulation)
training = np.array(training)
recall = np.array([100,100+delay_stimulation])
index=[range(0,N),range(N,N + N*N),range(N + N*N,N + N*N +N)]
               
               
               
# main function
def f(yt,t,tag,dt):
    global theta
    rLA = yt[index[0]][np.newaxis].T
    rLA = rLA*(rLA>1e-5)

    WLALA = yt[index[1]].reshape((N,N))
    exc =   yt[index[2]][np.newaxis].T
    
    INPUT_rCS = WCS0.dot(rCS(t).T)[np.newaxis].T
    
    rinhib = I0 + np.sum(rLA)*I1
        
    drLAdt = (-rLA + np.maximum(0,WLALA.dot(rLA) + INPUT_rCS - rinhib + exc ) )/taur
        
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
        if rLA[i] > threshold:
            if tag[i] == 0:
                tag[i] = t
        if tag[i] !=0:
            if t > tag[i] + max(training) - 100 and t < tag[i] + max(training):
                dexcdt[i] = (E + exc0[i] - exc[i]) / tauf
            else:
                dexcdt[i] = (exc0[i] - exc[i]) / taus
                


    dydt = np.concatenate((drLAdt.flatten(),
                           dWLALAdt.reshape((N*N)).flatten(),
                           dexcdt.flatten()))
    
    return dydt




# delays

np.disp('Starting overlap loop on 0 seeds and different delays')
Nrep = 0
delay_list = 3600*1000*np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28,30])
Overlap = np.zeros((Nrep,len(delay_list)))
exluded = np.zeros(Nrep)

for seed in range(0,Nrep):
	np.disp('Seed = '+str(seed))
	np.random.seed(seed+1)
	exc0 = np.random.normal(0,.5,N) 
	exc0 = np.sqrt(exc0*exc0)
	cc =0
	for delay in delay_list:
       
		# CS1 - CS2
		seqrCS1 = np.concatenate(( training, delay + od + recall ))+plus
		seqrCS2 = np.concatenate(( delay + training, delay + od + 25*60*1000 + recall ))+plus
		seqrCS3 =  []
		seqrUS = np.concatenate(( training, delay + training ))+plus
		T_list = [0,plus,max(training)+max(training)+plus, delay+plus , delay + max(training) + max(training)+plus, delay + od+plus, delay + od +1000+plus, delay + od + 25*60*1000+plus, delay + od + 25*60*1000 + 300+plus]
		dt_list = [.5,.5,20000,.5,20000,.5,20000,.5]
		rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)
		y = run(rCS,rUS,time_steps,f)
		rLA = y[index[0][:],:len(time_steps)]
		active = rLA > threshold2
		activeCS1recall = np.sum(active[:,int(np.where(time_steps == plus + delay + od + 100 )[0]):int(np.where(time_steps == plus + delay + od + 200)[0])], axis=1) >0
		activeCS2recall = np.sum(active[:,int(np.where(time_steps == plus + delay + od + 25*60*1000 + 100)[0]):int(np.where(time_steps == plus + delay + od + 25*60*1000 + 200)[0])], axis=1) >0
		Overlap[seed,cc] = sum(activeCS1recall*activeCS2recall) / sum(activeCS1recall) *100
		np.disp([delay/3600/1000,Overlap[seed,cc]])
		cc +=1
		if np.sum(np.sum(y[index[0][:]]>100))>0:
			exluded[seed] = 1
		
#with open('overlapLOOPdelay_01_08.pkl', 'wb') as f:
#	pickle.dump([Overlap], f)
	
#with open('excluded_overlapLOOPdelay_01_08.pkl', 'wb') as f:
#	pickle.dump([exluded], f)
	
	


# I0	
np.disp('Starting overlap loop on 0 seeds and 2 different I0')
Nrep = 0
I0_list = [5.8,6.2]
Overlap = np.zeros((Nrep,len(I0_list)))
exluded = np.zeros(Nrep)
y = []
for seed in range(0,Nrep):
	np.disp('Seed = '+str(seed))
	np.random.seed(seed+1)
	exc0 = np.random.normal(0,.5,N) 
	exc0 = np.sqrt(exc0*exc0)
       
	delay = 6*3600*1000
	cc = 0
	for I0 in I0_list:
       
		# CS1 - CS2
		seqrCS1 = np.concatenate(( training, delay + od + recall ))+plus
		seqrCS2 = np.concatenate(( delay + training, delay + od + 25*60*1000 + recall ))+plus
		seqrCS3 =  []
		seqrUS = np.concatenate(( training, delay + training ))+plus
		T_list = [0,plus,max(training)+max(training)+plus, delay+plus , delay + max(training) + max(training)+plus, delay + od+plus, delay + od +1000+plus, delay + od + 25*60*1000+plus, delay + od + 25*60*1000 + 300+plus]
		dt_list = [.5,.5,20000,.5,20000,.5,20000,.5]
		rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)
		y = run(rCS,rUS,time_steps,f)
		rLA = y[index[0][:],:len(time_steps)]
		active = rLA > threshold2
		activeCS1recall = np.sum(active[:,int(np.where(time_steps == plus + delay + od + 100 )[0]):int(np.where(time_steps == plus + delay + od + 200)[0])], axis=1) >0
		activeCS2recall = np.sum(active[:,int(np.where(time_steps == plus + delay + od + 25*60*1000 + 100)[0]):int(np.where(time_steps == plus + delay + od + 25*60*1000 + 200)[0])], axis=1) >0
		Overlap[seed,cc] = sum(activeCS1recall*activeCS2recall) / sum(activeCS1recall) *100
		np.disp([delay/3600/1000,Overlap[seed,cc]])
		cc +=1
		if np.sum(np.sum(y[index[0][:]]>100))>0:
			exluded[seed] = 1
		
#with open('overlapLOOPI0_01_08.pkl', 'wb') as f:
#	pickle.dump([Overlap], f)
	
#with open('excluded_overlapLOOPI0_01_08.pkl', 'wb') as f:
#	pickle.dump([exluded], f)
	

# E
np.disp('Starting overlap loop on 20 seeds and 2 different E')
Nrep = 20
E_list = [3]
Overlap = np.zeros((Nrep,len(E_list)))
exluded = np.zeros(Nrep)

for seed in range(0,Nrep):
	np.disp('Seed = '+str(seed))
	np.random.seed(seed+1)
	exc0 = np.random.normal(0,.5,N) 
	exc0 = np.sqrt(exc0*exc0)
       
	delay = 6*3600*1000
	I0 = 6
       
	cc =0
	for E in E_list:
       
		# CS1 - CS2
		seqrCS1 = np.concatenate(( training, delay + od + recall ))+plus
		seqrCS2 = np.concatenate(( delay + training, delay + od + 25*60*1000 + recall ))+plus
		seqrCS3 =  []
		seqrUS = np.concatenate(( training, delay + training ))+plus
		T_list = [0,plus,max(training)+max(training)+plus, delay+plus , delay + max(training) + max(training)+plus, delay + od+plus, delay + od +1000+plus, delay + od + 25*60*1000+plus, delay + od + 25*60*1000 + 300+plus]
		dt_list = [.5,.5,20000,.5,20000,.5,20000,.5]
		rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)
		y = run(rCS,rUS,time_steps,f)
		rLA = y[index[0][:],:len(time_steps)]
		active = rLA > threshold2
		activeCS1recall = np.sum(active[:,int(np.where(time_steps == plus + delay + od + 100 )[0]):int(np.where(time_steps == plus + delay + od + 200)[0])], axis=1) >0
		activeCS2recall = np.sum(active[:,int(np.where(time_steps == plus + delay + od + 25*60*1000 + 100)[0]):int(np.where(time_steps == plus + delay + od + 25*60*1000 + 200)[0])], axis=1) >0
		Overlap[seed,cc] = sum(activeCS1recall*activeCS2recall) / sum(activeCS1recall) *100
		np.disp([delay/3600/1000,Overlap[seed,cc]])
		cc +=1
		if np.sum(np.sum(y[index[0][:]]>100))>0:
			exluded[seed] = 1
		
with open('overlapLOOPE_01_08.pkl', 'wb') as f:
	pickle.dump([Overlap], f)
	
with open('excluded_overlapLOOPE_01_08.pkl', 'wb') as f:
	pickle.dump([exluded], f)
