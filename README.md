# QueuingBandits

__Code for Regret of Queuing Bandits__

Paper url: https://arxiv.org/abs/1604.06377 

Dependencies: pandas, numpy, matplotlib, multiprocessing

Function to simulate QB for many runs in parallel: ep_greedy_avg_wopt_par_var(mu,u,K,l,t,num_sim,nthread):
	
  
  ```python
  '''
     num_sim: number of experiments
	   nthread: number of parallel threads (recommended to number of cores in the machine)
	   mu: True service rate matrix (u * K numpy array : all elements less than 1.0)
	   u: number of users
	   K: number of servers
	   l: service rates (numpy array of size u)
	 '''

