
import numpy as np
from numpy.random import random_sample
import random
from math import log
from multiprocessing import Pool
from mwmatching import maxWeightMatching
import matplotlib.pyplot as plt


def bipartiteMatch(graph):
	'''Find maximum cardinality matching of a bipartite graph (U,V,E).
	The input format is a dictionary mapping members of U to a list
	of their neighbors in V.  The output is a triple (M,A,B) where M is a
	dictionary mapping members of V to their matches in U, A is the part
	of the maximum independent set in U, and B is the part of the MIS in V.
	The same object may occur in both U and V, and is treated as two
	distinct vertices if this happens.'''
	
	# initialize greedy matching (redundant, but faster than full search)
	matching = {}
	for u in graph:
		for v in graph[u]:
			if v not in matching:
				matching[v] = u
				break
	
	while 1:
		# structure residual graph into layers
		# pred[u] gives the neighbor in the previous layer for u in U
		# preds[v] gives a list of neighbors in the previous layer for v in V
		# unmatched gives a list of unmatched vertices in final layer of V,
		# and is also used as a flag value for pred[u] when u is in the first layer
		preds = {}
		unmatched = []
		pred = dict([(u,unmatched) for u in graph])
		for v in matching:
			del pred[matching[v]]
		layer = list(pred)
		
		# repeatedly extend layering structure by another pair of layers
		while layer and not unmatched:
			newLayer = {}
			for u in layer:
				for v in graph[u]:
					if v not in preds:
						newLayer.setdefault(v,[]).append(u)
			layer = []
			for v in newLayer:
				preds[v] = newLayer[v]
				if v in matching:
					layer.append(matching[v])
					pred[matching[v]] = v
				else:
					unmatched.append(v)
		
		# did we finish layering without finding any alternating paths?
		if not unmatched:
			unlayered = {}
			for u in graph:
				for v in graph[u]:
					if v not in preds:
						unlayered[v] = None
			return (matching,list(pred),list(unlayered))

		# recursively search backward through layers to find alternating paths
		# recursion returns true if found path, false otherwise
		def recurse(v):
			if v in preds:
				L = preds[v]
				del preds[v]
				for u in L:
					if u in pred:
						pu = pred[u]
						del pred[u]
						if pu is unmatched or recurse(pu):
							matching[v] = u
							return 1
			return 0

		for v in unmatched: recurse(v)
            

def flatten(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result            
            
            
            
            
def max_match(U,u,K):
    #maximal cardinality optimal matching
    dic = {}
    for i in range(u):
        row = U[i,:]
        index = row.argmax()
        dic[i] = [index]

    (match,A,B) = bipartiteMatch(dic)
    unmatched = [item for item in range(u) if item not in match.values()]

    if len(match) != u:
        for i in unmatched:
            row = U[i,:]
            indices = np.argsort(row).tolist()
            indices = flatten(indices)
            for j in range(K):
                if indices[K-1-j] not in match:
                    match[indices[K-1-j]] = i
                    break
    #print match

    P = np.zeros([u,K])
    for i in match:
        P[match[i],i] = 1
    return np.matrix(P)




def one_schedule(P,mu,u,K,q,N,ucb_avg,l):
	#one scheduling step
	#P : permutation matrix
	#mu : actual transmission rate
	#K : number of queues
	#q : current queuelengths
	#N : number of samplings to be updates
	#avg : number of samplings
    rates  = np.diag(P*np.transpose(mu))
    for i in range(u):
        bit = random.random()
        if bit <= l[i]:
            q[i] = q[i] + 1 
    nodes = P*np.transpose(np.matrix(range(K)))
    for i in range(u):
        bit = random.random()
        if bit <= rates[i]:
            #print "testing"
            if q[i] != 0 :
                q[i] = q[i] - 1
            ucb_avg[i,int(nodes[i])] = (N[i,int(nodes[i])]*ucb_avg[i,int(nodes[i])] + 1)/(N[i,int(nodes[i])]+1)
            N[i,int(nodes[i])] = N[i,int(nodes[i])] + 1
        else:
            ucb_avg[i,int(nodes[i])] = (N[i,int(nodes[i])]*ucb_avg[i,int(nodes[i])])/(N[i,int(nodes[i])]+1)
            N[i,int(nodes[i])] = N[i,int(nodes[i])] + 1
            
def stat_sample(lam,mu,num,tol):
	values = range(tol)
	prob = [0]*tol
	prob[0] = 1 - lam/mu
	prob[1] = lam*(mu - lam)/((mu**2)*(1 - lam))
	s = prob[0] + prob[1]
	for i in range(2,tol):
		prob[i] = prob[1]*(lam*(1-mu)/(mu*(1 - lam)))**(i-1)
		s = s + prob[i]
	for i in range(tol) :
		prob[i] = prob[i]/s
	#print values
	#print prob
	return np.random.choice(values,size = num, p = prob)[0]#weighted_values(values,prob,num)



def ep_greedy_wopt(mu,u,K,l,t):
	np.random.seed()
	random.seed()
 	q = [0]*u
 	q_o = [0]*u
	N = np.zeros([u,K])
	ucb_avg = np.zeros([u,K])
	N_o = np.zeros([u,K])
	ucb_avg_o = np.zeros([u,K])
	for i in range(u):
		mui = np.amax(mu[i,:])
		lam = l[i]
		q[i] = stat_sample(lam,mui,1,200)
		q_o[i] = q[i]
	queue = np.empty((0,u),int)
	queue_o = np.empty((0,u),int)
	queue = np.append(queue, np.array([q]), axis=0)
	queue_o = np.append(queue_o, np.array([q_o]), axis=0)
	P_o = max_match(mu,u,K)
	P_exp = []
	for i in range(K):   #these are the explore schedules
		P = []
		for j in range(u):
			h = [0]*K
			h[(j+i)%K] = 1
			P = P + [h]
		P = np.matrix(P)
		P_exp = P_exp + [P]	
	s = 0
	for i in range(K):
		one_schedule(P_exp[i],mu,u,K,q,N,ucb_avg,l)
		queue = np.append(queue, np.array([q]), axis=0)
		one_schedule(P_o,mu,u,K,q_o,N_o,ucb_avg_o,l)
		queue_o = np.append(queue_o, np.array([q_o]), axis=0)
		s = s + 1

	while (s < t):
		bit = random.random()
		if (bit <= float(3*K*(log(s)**2)/s) ):
		#if (bit <= float(3*K*(log(s)**3)/s) ):
		#if (bit <= float(4*K/s) ):
		#if bit > 1:
			values = range(K)
			prob = [float(1./K)]*K
			index = np.random.choice(range(K))
			one_schedule(P_exp[index],mu,u,K,q,N,ucb_avg,l)
			s = s + 1
			queue = np.append(queue, np.array([q]), axis=0)
		else:
			ucb_mat = np.zeros([K,K])
			for i in range(u):
				for j in range(K):
					ucb_mat[i,j] = ucb_avg[i,j] + ((log(s)**2)/(2*N[i,j]))**0.5 
					#ucb_mat[i,j] = ucb_avg[i,j] + ((log(s)**3)/(2*N[i,j]))**0.5 
			P_s = max_match(ucb_mat,u,K)            
 			one_schedule(P_s,mu,u,K,q,N,ucb_avg,l)
 			s = s+1
 			queue = np.append(queue, np.array([q]), axis=0)
 		one_schedule(P_o,mu,u,K,q_o,N_o,ucb_avg_o,l)
		queue_o = np.append(queue_o, np.array([q_o]), axis=0)




 	return queue,queue_o


def ep_helper(sim):
	'''Helper Function '''
	return ep_greedy_wopt(sim[0],sim[1],sim[2],sim[3],sim[4])

def ep_greedy_avg_wopt_par_var(mu,u,K,l,t,num_sim,nthread):
	'''Run in parallel
	   num_sim: number of experiments
	   nthread: number of parallel threads (recommended to number of cores in the machine)
	   mu: True service rate matrix (u * K numpy array : all elements less than 1.0)
	   u: number of users
	   K: number of servers
	   l: service rates (numpy array of size u)
	 '''
 	pool = Pool(processes=nthread)
 	result = pool.map(ep_helper,[(mu,u,K,l,t)]*num_sim)
 	cleaned = [x for x in result if not x is None] # getting results
 	#cleaned = asarray(cleaned)
 	pool.close() # not optimal! but easy
 	pool.join()
 	(Q,Q_o) = cleaned[0]
 	Q = Q.astype(float)
 	Q_o = Q_o.astype(float)
 	vQ = np.square(Q - Q_o)
 	for i in range(1,len(cleaned)):
 		(x,y) = cleaned[i]
 		Q = Q + x.astype(float)
 		Q_o = Q_o+y.astype(float)
 		vQ = vQ + np.square(x.astype(float)-y.astype(float))
 		

 	Q = Q/num_sim
 	Q_o = Q_o/num_sim
 	vQ = vQ/(num_sim - 1)
 	sQ = abs(vQ - np.square(Q - Q_o))
 	sQ = np.sqrt(sQ)
 	return (Q,Q_o,sQ)

