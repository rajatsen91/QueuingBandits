from src.QB import *


if __name__=='__main__':

	u = 3 #number of users
	K = 5 #number of servers
	mu = np.matrix([[0.5,0.33,0.33,0.33,0.25],[0.33,0.5,0.25,0.33,0.25],[0.25,0.33,0.5,0.25,0.25]]) #u * K service rate matrix 
	l = [0.35,0.35,0.35] # list of arrival rates
	t = 10000 #time steps
	num_sim = 1000 #number of simulations
	(q,q0,sq) = ep_greedy_avg_wopt_par_var(mu,u,K,l,t,num_sim,32) #q - q0: queue regret, sq: 2*std. deviation of q-regret

	x = q - q0

	np.save('./results/qb_ep_3_5_17_15.npy',x) #saving regret for all the u queues
	np.save('./results/sqb_ep_3_5_17_15.npy',sq) #saving std. for all the u queues  3*t dimensional arrays