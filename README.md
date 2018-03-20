# QueuingBandits

__Code for Regret of Queuing Bandits__

Paper url: https://arxiv.org/abs/1604.06377 

Dependencies: pandas, numpy, matplotlib, multiprocessing

Function to simulate QB for many runs in parallel: ep_greedy_avg_wopt_par_var(mu,u,K,l,t,num_sim,nthread)
	
  
  ```python
  '''
     	   num_sim: number of experiments
	   nthread: number of parallel threads (recommended to number of cores in the machine)
	   mu: True service rate matrix (u * K numpy matrix : all elements less than 1.0)
	   u: number of users
	   K: number of servers
	   l: service rates (listof size u)
	   t: Time Horizon
	   
	   Returns: (q,q0,sq)
	   q: array of average queue lengths for bandit algorithm for u queues each till time horizon t
	   q0: array of average optimal queue length for u queues each till time horizon t
	   sq: std. of queue regret (q - q0) over all the runs
	 '''
	 
```

__Example File__

1. An example run file has been provided in ./examples/ folder. To run the example files clone the repo. Go to the directory of the repo. Then run the example file:

```
python ./examples/Q_run_ep.py

```
2. The output will be stored as two .npy files in the ./results/ directory. 




