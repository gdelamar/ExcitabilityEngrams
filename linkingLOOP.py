from numpy import loadtxt
import numpy as np
import random
from numpy import linalg as LA
import time
import pickle

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
    


np.disp('Running linking LOOP')
# np.random.seed(5)


# 23/06
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
I0 = 6
I1 = .9
threshold = 4
threshold2 = threshold
E = 4
tauf =  taur
taus =  12 * 3600 * 1000
CS = 4
US = 1
extra = 5
Nextra = 8
Ncut = 6
current1set = .5
current2set = .5
I1min = .89
I0min = I0
plus = 2000
od = 24 * 3600 * 1000

global theta
theta = 0
Nstim = 15
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








h5 = 5 * 3600 * 1000
d7 = 7 * 24 * 3600 * 1000
d2 = 2 * 24 * 3600 * 1000

Nrep = 20
USlist = [0,.1,.2,.3,.4,.5,.6,.7]
delaylist = 3600*1000*np.array([3,4,5,6,7,8])

freezing = np.zeros([Nrep,2*len(USlist)*len(delaylist)])
M = max(training)


for seed in range(0,Nrep,1):


	cc = 0

	np.disp('Seed = '+str(seed))
	
	for US in USlist:
		for delay in delaylist:

			#################################
			np.random.seed(seed)

		    
		    
		    
			# initialisation
			rLA0 = np.zeros(N)
			WLALA0 = np.zeros(N*N)
			exc0 = np.random.normal(0,.5,N) 
			exc0 = np.sqrt(exc0*exc0)

			h5 = 5 * 3600 * 1000
			d7 = 7 * 24 * 3600 * 1000
			d2 = 2 * 24 * 3600 * 1000

			M = max(training)

			T_list = [0,plus,max(training)+max(training)+plus, plus+ d7 , plus+ d7 + max(training) + max(training), plus+ d7 + delay, plus+ d7 + delay + max(training) + max(training), plus+ d7 + delay + d2, plus+ d7 + delay + d2 + max(training) + max(training),plus+ d7 + delay + d2 + d2,plus+ d7 + delay + d2 + d2 + 400]
			dt_list = [.5,.5,20000,.5,20000,.5,20000,.5,20000,.5]



			# Recall CS2
			seqrCS1 = []
			seqrCS2 = np.concatenate(( plus+training + d7, plus+recall + d7 + delay + d2 + d2 ))
			seqrCS3 = np.concatenate(( plus+training + d7 + delay, plus+training + d7 + delay + d2))
			seqrUS = plus+training + d7 + delay + d2


			rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)
			y2 = run(rCS,rUS,time_steps)


			# Observer
			recCS2 = y2[index[0][:],int(np.where(time_steps ==  plus + d7 + delay + d2 + d2 +100 )[0]):int(np.where(time_steps == plus + d7 + delay + d2 + d2 + 200 )[0])]







			tcut = int(np.where(time_steps == plus + d7 + delay + d2 + d2)[0][0])

			rLA0 = y2[index[0]][:,tcut]
			WLALA0 = y2[index[1]][:,tcut]
			exc0 = y2[index[2]][:,tcut]

	
			# Recall CS3
			seqrCS1 = []
			seqrCS2 = []
			seqrCS3 = recall
			seqrUS = []

			T_list = [0,500]
			dt_list = [1]
			rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)
			y3 = run(rCS,rUS,time_steps)

			# Observer
			recCS3 = y3[index[0][:],int(np.where(time_steps ==  100 )[0]):int(np.where(time_steps == 200 )[0])]

			T_list = [0,plus,max(training)+2000+plus, plus+ d7 , plus+ d7 + max(training) + 2000, plus+ d7 + h5, plus+ d7 + h5 + max(training) + 2000, plus+ d7 + h5 + d2, plus+ d7 + h5 + d2 + max(training) + 2000,plus+ d7 + h5 + d2 + d2,plus+ d7 + h5 + d2 + d2 + 300]
			dt_list = [.5,.5,20000,.5,20000,.5,20000,.5,20000,.5]
			rCS, rUS, time_steps = protocolUS(seqrCS1,seqrCS2,seqrCS3,seqrUS,T_list,dt_list)


			freezing[seed,2*cc] = max(0,np.sum(np.mean(recCS2,axis=1)))
			freezing[seed,2*cc+1] = max(0,np.sum(np.mean(recCS3,axis=1)))

			np.disp([US,delay/3600/1000,freezing[seed,2*cc],freezing[seed,2*cc+1]])


			
			cc+=1
    			




    
    
with open('freezing_linkingLOOP_30_06.pkl', 'wb') as f:
	pickle.dump([freezing], f)
	
    
