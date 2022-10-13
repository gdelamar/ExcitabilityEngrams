from numpy import loadtxt
import numpy as np
import random
from numpy import linalg as LA
import time
import pickle

np.disp('Starting competition on 10 seeds and different current1 and Ncut')

def run(rCS,rUS, time_steps):
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
    M = max(training)

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



######### set of parameters ############
# 25/05 back to 17/05
np.random.seed(1)
N=60
NCS=30
taur=20
tauw=600
zetaLA=0
rLA0 = np.zeros(N)
WLALA0 = np.zeros(N*N)
Wlow = .21
WCS0 = np.zeros((N,NCS)) + Wlow
WCS0[:int(N/4),:int(NCS/3)] = .3
WCS0[int(N/4):int(N/2),int(NCS/3):2*int(NCS/3)] = .3
WCS0[int(N/2):3*int(N/4),2*int(NCS/3):] = .3
I0 = 3
I1 = .9
threshold = 3
E = 1.5
tauf =  taur
taus =  12 * 3600 * 1000
exc0 = np.random.normal(0,.5,N) 
exc0 = np.sqrt(exc0*exc0)
CS = 2.5
US = 1
plus = 2000
od = 24 * 3600 * 1000
Nstim = 20
delay_between_event = 150
delay_stimulation = 40
extra = 1
Nextra = 6
Ncut = 6
current1set = .6
current2set = current1set
I1min = .88
I0min = I0

# 27/05
np.random.seed(7)
N=60
NCS=30
taur=20
tauw=600
zetaLA=0
rLA0 = np.zeros(N)
WLALA0 = np.zeros(N*N)
Wlow = .21
WCS0 = np.zeros((N,NCS)) + Wlow
WCS0[:int(N/4),:int(NCS/3)] = .3
WCS0[int(N/4):int(N/2),int(NCS/3):2*int(NCS/3)] = .3
WCS0[int(N/2):3*int(N/4),2*int(NCS/3):] = .3
I0 = 3
I1 = .9
threshold = 3
E = 1.5
tauf =  taur
taus =  12 * 3600 * 1000
exc0 = np.random.normal(0,.5,N) 
exc0 = np.sqrt(exc0*exc0)
CS = 2.5
US = 1
plus = 2000
od = 24 * 3600 * 1000
Nstim = 20
delay_between_event = 150
delay_stimulation = 40
extra = 1
Nextra = 6
Ncut = 6
current1set = .3
current2set = .3
I1min = .88
inhib = False

# 30/05
N=60
NCS=30
taur=20
tauw=500
zetaLA=0
rLA0 = np.zeros(N)
WLALA0 = np.zeros(N*N)
Wlow = .2
WCS0 = np.zeros((N,NCS)) + Wlow
WCS0[:int(N/4),:int(NCS/3)] = .3
WCS0[int(N/4):int(N/2),int(NCS/3):2*int(NCS/3)] = .3
WCS0[int(N/2):3*int(N/4),2*int(NCS/3):] = .3
I0 = 3
I1 = .9
threshold = 3
threshold2 = threshold
E = 1.5
tauf =  taur
taus =  12 * 3600 * 1000
CS = 2.5
extra = 3
Nextra = 7
US = 1
current2set = .1
Ncut = 7
current1set = .1
I1min = .92
I0min = I0
plus = 2000
od = 24 * 3600 * 1000


#31/05
N=60
NCS=30
taur=20
tauw=600
zetaLA=0
rLA0 = np.zeros(N)
WLALA0 = np.zeros(N*N)
Wlow = .2
WCS0 = np.zeros((N,NCS)) + Wlow
WCS0[:int(N/4),:int(NCS/3)] = .3
WCS0[int(N/4):int(N/2),int(NCS/3):2*int(NCS/3)] = .3
WCS0[int(N/2):3*int(N/4),2*int(NCS/3):] = .3
I0 = 3.5
I1 = .9
threshold = 4
threshold2 = threshold
E = 2
tauf =  taur
taus =  12 * 3600 * 1000
CS = 3
US = 1
extra = 3
Nextra = 7
Ncut = 7
current1set = .2
current2set = .2
I1min = .88
I0min = I0
plus = 2000
od = 24 * 3600 * 1000

# 01/06
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
current1set = 1
current2set = 1
I1min = .88
I0min = I0
plus = 2000
od = 24 * 3600 * 1000 

# 02/06
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
current1set = 2
current2set = 2
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000 


# 06/06
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
current1set = .5
current2set = .5
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000 


# 10/06
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
current1set = 2
current2set = 2
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000 

# 17/06
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
E = 4
tauf =  taur
taus =  12 * 3600 * 1000
CS = 4
US = 1
extra = 4
Nextra = 8
Ncut = 6
current1set = .5
current2set = .5
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000
sigma = .5

# 18/06
N=60
NCS=30
taur=20
tauw= 700
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
E = 4
tauf =  taur
taus =  12 * 3600 * 1000
CS = 4
US = 1
extra = 4
Nextra = 9
Ncut = 7
current1set = .5
current2set = .5
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000


# 21/06
N=60
NCS=30
taur=20
tauw= 700
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
Nextra = 8
Ncut = 6
current1set = .5
current2set = .5
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000


# 23/06
N=60
NCS=30
taur=20
tauw= 750
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
E = 4
tauf =  taur
taus =  12 * 3600 * 1000
CS = 4
US = 1
extra = 4
Nextra = 6
Ncut = 4
current1set = 2
current2set = 2
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000

# 24/06
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
Nextra = 8
Ncut = 6
current1set = .5
current2set = .5
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000


# 25/06, set 19/06 that works well on bias alloc, overlap and competition
# modified
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
extra = 3
Nextra = 10
Ncut = 7
current1set = .5
current2set = .5


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
    
    if t > plus + delay and t < plus + delay + max(training) and inhib:
        rinhib = I0min + np.sum(rLA)*I1min
    else:
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






#L_current1 = [0,.1,.2,.3,.4,.5]
#L_current1 = [.6,.7,.8,.9,1]
#L_current1 = [.2]
#freezing2 = np.zeros((np.size(L_current1),4))

Nrep = 20
L_current1 = [.1,.2,.3,.4,.5]
L_Ncut = [8]
Ncurrent1 = len(L_current1)
NNcut = len(L_Ncut)

freezing2 = np.zeros((Nrep,4*Ncurrent1*NNcut))
freezing2bis = np.zeros((Nrep,4*Ncurrent1*NNcut))
current1 = current1set
loop = 0

for seed in range(0,Nrep):
	np.random.seed(seed+1)
	exc0 = np.random.normal(0,sigma,N) 
	exc0 = np.sqrt(exc0*exc0)
	cc = 0
	for Ncut in L_Ncut:
		for current1 in L_current1:
			np.disp("Seed = "+str(seed)+", current1 = "+str(current1)+", Ncut = "+str(Ncut))
			for inhib in [False,True]:
				#for current2 in [0,current2set]:		
				for current2 in [0]:
					for delay in [6*3600 *1000,24*3600 *1000]:  
					#for delay in [6*3600 *1000]:  

						# Parameters
						seqrCS1 = np.concatenate(( training, delay + od + recall ))+plus
						seqrCS2 = np.concatenate(( delay + training, delay + od + 25*60*1000 + recall ))+plus
						seqrCS3 =  []
						seqrUS = np.concatenate(( training, delay + training ))+plus
						T_list = [0,plus,max(training)+max(training)+plus, delay+plus , delay + max(training) + max(training)+plus, delay + od+plus, delay + od +1000+plus, delay + od + 25*60*1000+plus, delay + od + 25*60*1000 + 300+plus]
						dt_list = [.5,.5,20000,.5,20000,.5,20000,.5]
						rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)

						def block(t):
							L1 = range(Ncut) # inhibit only a subset of neurons
							L2 = range(Ncut) # inhibit only a subset of neurons

							Lneurons1 = np.zeros(N)
							Lneurons2 = np.zeros(N)

							for i in L1:
								Lneurons1[i] = 1
							for i in L2:
								Lneurons2[i] = 1
							if (t > plus + delay and t < plus + delay + max(training)):
								return - Lneurons1 * current1
							if (t > plus + delay + od):
								return - Lneurons2 * current2
							else:
								return np.zeros(N)

						# Run
						y = run(rCS,rUS,time_steps)



						# Compute freezing
						#recCS1 = y[index[0][:],np.where(time_steps == plus + delay + od + 100)[0][0]:np.where(time_steps == plus + delay + od + 200)[0][0]]
						recCS2 = y[index[0][:],np.where(time_steps == plus + delay + od + 25*60*1000 + 100)[0][0]:np.where(time_steps == plus + delay + od + 25*60*1000 + 200)[0][0]]


						freezing2[loop,cc] = np.sum(np.mean(np.maximum(0,recCS2),axis = 1))
						freezing2bis[loop,cc] = np.sum(np.mean(np.maximum(0,recCS2 - threshold), axis = 1))
						np.disp([current2, Ncut, delay/3600/1000,freezing2[loop,cc],freezing2bis[loop,cc]])		
						cc+=1
	loop +=1
    
    
with open('freezing2_competitionLOOP_24_06.pkl', 'wb') as f:
	pickle.dump([freezing2], f)
with open('freezing2bis_competitionLOOP_24_06.pkl', 'wb') as f:
	pickle.dump([freezing2bis], f)



