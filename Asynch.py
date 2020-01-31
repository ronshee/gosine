# -*- coding: utf-8 -*-
"""
The Gossiping Insert-Eliminate (GoSInE) Algorithm: Asynchronous Version
@author: Ronshee
"""

import numpy as np
import matplotlib.pyplot as plt

def stickyarms(n, K): # Initializing the set maintained by the agents
    arms = np.zeros((np.int(np.ceil(K/n)+2), n))
    for p in range(n):
        for q in range(np.int(np.ceil(K/n)+2)):
            if (np.mod(p*np.ceil(K/n),K)+q+1 > K):
                arms[q,p] = np.mod(np.mod(p*np.ceil(K/n),K)+q+1-K,K)
            else:
                arms[q,p] = np.mod(p*np.ceil(K/n),K)+q+1
    arms = arms - 1
    return arms.astype(np.int64)
# Structure of the set maintained by the agents:
# Columns denote the agents, where each agent has ceil(K/n)+2 arms, thus ceil(K/n)+2 rows.
# The first ceil(K/n) arms maintained by every agent constitute its sticky set,
# which remains fixed at all times.
# The ceil(K/n)+1 th arm is U and ceil(K/n)+2 th arm is L.
# U is updated by choosing the most played arm among U and L in a phase.
# L is the least played arm in previous phase, which will be replaced by the recommendation pulled by the agent
# at the end of a phase.

def ucb(mu, alpha, ucb_score, mu_bar, T, idx, s): # Subroutine which plays UCB for the chosen arm
    T[idx]=T[idx]+1             # Number of times the arm being played incremented by 1
    if np.random.rand() <= mu[idx]:         # When a reward of 1 is received after playing the arm
        mu_bar[idx]=(((T[idx]-1)*mu_bar[idx])+1)/T[idx]
    else:           # When a reward of 0 is received after playing the arm
        mu_bar[idx]=((T[idx]-1)*mu_bar[idx])/T[idx]
    regret=np.sum((mu[0]-mu)*T)         # Regret incurred
    ucb_score=mu_bar+np.sqrt(alpha*np.log(s+1)/T);  # Updating UCB score for next time step
    return regret, ucb_score, mu_bar, T

#Note: this getarm function routine takes the adjacency matrix of the graph as input
#and works only for the case when the neighbors of the agent are chosen equally likely.
#For a general gossip matrix P, this getarm function has to be modified
#to pick the agent sampled from P(.,ag) from where ag pulls a recommendation 
def getarm(arms, neighbor, ag, delta_T):
    nb=np.nonzero(neighbor[:,ag])[0]    # neighbors of agent ag
    rec_ag=nb[np.random.randint(np.size(nb))]   # picking the neighbor whose best arm recommendation will be pulled by ag
    rec_id=arms[np.argmax(delta_T[arms[:,rec_ag],rec_ag]),rec_ag]   #  estimated best arm of the neighbor picked by ag
    return rec_id        

n = 15        # number of agents
K = 40         # number of arms
m = 40         # number of phases
alpha = 4      # UCB parameter
dl = 0.5          # Phase length slack
itr = 100        # Number of runs over the same instance for confidence plots
mu=0.85*np.random.rand(K)    # mean reward vector
mu[::-1].sort()     # sort the mean vector so that arm 1 is the best arm
mu[0]=0.95  # mean reward of best arm
mu[1]=0.85  # mean reward of second best arm
t = m**3    # Time horizon
rs=np.zeros((t,itr))  # regret for separate single agent playing the same MAB instance over all itr runs
rpa_comp=np.zeros((t,itr))  # regret per agent for complete graph over all itr runs

# adjacency matrix of complete graph
neighbor_comp=np.ones((n,n),int)    
np.fill_diagonal(neighbor_comp, 0)

for l in range(itr):
    j=np.ones(n)    # to keep track of which phase every agent is in separately for each of them (since asynchronous)
    phaselen=np.random.randint(j[0]**3,np.ceil((1+dl)*(j[0]**3))+1,size=n)  # initialzing the time when the first phase ends for every agent
    ucb_score_single=np.Inf*np.ones(K)  # ucb score of a separate single agent playing the same MAB instance
    T_single=np.zeros(K)  # number of times an arm is played by the separate single agent playing the same MAB instance
    mu_bar_single=np.zeros(K)  # average mean reward of the arms estimated by the separate single agent playing the same MAB instance
    regret_single=np.zeros(t) # regret incurred by the separate single agent playing the same MAB instance
    mu_bar_comp=np.zeros((K,n)) # average mean reward of the arms estimated by the agents connected by a complete graph
    T_comp=np.zeros((K,n)) # number of times an arm is played by the agents connected by a complete graph
    T_prev_comp=np.zeros((K,n)) # number of times an arm is played by the agents connected by a complete graph till the previous phase
    delta_T_comp=np.zeros((K,n)) # to keep track of how many times each arm is played by every agent in the previous phase
    ucb_score_comp=np.Inf*np.ones((K,n))    # UCB scores of arms for agents connected by a complete graph
    regret_comp=np.zeros((t,n)) # Regret incurred by agents connected by a complete graph
    arms_comp = stickyarms(n, K) # Initializing the set maintained by agents connected by a complete graph
        
    for s in range(t):
        # UCB played by the separate single agent playing the same MAB instance
        idx_single=np.argmax(ucb_score_single)  # Selecting the arm with highest UCB score
        regret_single[s], ucb_score_single, mu_bar_single, T_single=ucb(mu, alpha, ucb_score_single, mu_bar_single, T_single, idx_single, s+1)
        # playing UCB with the arm chosen above
        
        # Multi-Agent UCB on complete graph
        for ag in range(n):
            idx_comp=arms_comp[np.argmax(ucb_score_comp[arms_comp[:,ag],ag]),ag] # Selecting the arm with highest UCB score among the set of arms maintained
            regret_comp[s,ag], ucb_score_comp[:, ag], mu_bar_comp[:, ag], T_comp[:, ag]=ucb(mu, alpha, ucb_score_comp[:,ag], mu_bar_comp[:,ag], T_comp[:,ag], idx_comp, s+1)
            # playing UCB with the arm chosen above

            if s==phaselen[ag]-1:   # to check whether agent ag has reached the end of phase
                delta_T_comp[:,ag]=T_comp[:,ag]-T_prev_comp[:,ag]   # calculating number of times an arm is played by agent ag in the previous phase
                
                # Checking which arm out of U and L is played the least number of times and swapping them
                if (delta_T_comp[arms_comp[int(np.ceil(K/n)),ag],ag]<delta_T_comp[arms_comp[int(np.ceil(K/n)+1),ag],ag]):
                    tmp=arms_comp[int(np.ceil(K/n)),ag]
                    arms_comp[int(np.ceil(K/n)),ag]=arms_comp[int(np.ceil(K/n)+1),ag]
                    arms_comp[int(np.ceil(K/n)+1),ag]=tmp
                rec_id_comp=getarm(arms_comp,neighbor_comp,ag,delta_T_comp) # Recommendation received after pulling from one of the neighbors 
                if (np.size(np.where(arms_comp[:,ag]==rec_id_comp))==0): # Checking whether the received recommendation is in the set of the arms maintained by ag or not
                    arms_comp[int(np.ceil(K/n))+1,ag]=rec_id_comp # if not present, replace it with L
                T_prev_comp[:,ag]=np.copy(T_comp[:,ag]) # Stores the number of times an arm was played till the previous phase by all agents
                j[ag]=j[ag]+1 # updating the phase for agent ag
                phaselen[ag]=phaselen[ag]+np.random.randint((j[ag]**3)-((j[ag]-1)**3),np.ceil((1+dl)*((j[ag]**3)-((j[ag]-1)**3)))+1) # Updating the time when next phase ends for ag
    
    rs[:,l]=np.copy(regret_single)
    rpa_comp[:,l]=np.mean(regret_comp,axis=1) # per agent regret for lth run

# computing mean and confidences for the regret incurred over time t over itr runs
rs_avg=np.mean(rs,axis=1)
rs_lower=np.percentile(rs,2.5,axis=1)
rs_upper=np.percentile(rs,97.5,axis=1)
rpa_comp_avg=np.mean(rpa_comp,axis=1)
rpa_comp_lower=np.percentile(rpa_comp,2.5,axis=1)
rpa_comp_upper=np.percentile(rpa_comp,97.5,axis=1)

x=np.arange(t)

# Plotting
plt.plot(x,rs_avg,label='No Communication',color='b')
plt.fill_between(x,rs_lower,rs_upper,color='b')
plt.plot(x,rpa_comp_avg,label='Proposed Algorithm on Complete Graph',color='g')
plt.fill_between(x,rpa_comp_lower,rpa_comp_upper,color='g')
plt.xlabel('Time Horizon')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()