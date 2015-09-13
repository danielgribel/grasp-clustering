#!/usr/bin/env python

# summary: grasp algorithm for clustering.
# based on "Investigation of a new GRASP-based clustering algorithm applied to biological data" paper

# paper author: Maria C.V. Nascimento, Franklina M.B. Toledo, Andre C.P.L.F. de Carvalho
# code author: Daniel Lemes Gribel (dgribel@inf.puc-rio.br)

import numpy
import random
import networkx as nx
import time
from numpy.linalg import norm
from scipy.spatial import distance
from heapq import nlargest, nsmallest
from collections import Counter

# reading input data
def load_data(inputdata):
	mydata = numpy.loadtxt(inputdata, dtype = numpy.object, delimiter = ',')
	return mydata

def draw_graph(G):
	# an example using Graph as a weighted network.
	# __author__ = """Aric Hagberg (hagberg@lanl.gov)"""
	try:
	    import matplotlib.pyplot as plt
	except:
	    raise

	elarge = [(u,v) for (u,v,d) in G.edges(data = True) if d['weight'] > 0.5]
	esmall = [(u,v) for (u,v,d) in G.edges(data = True) if d['weight'] <= 0.5]

	pos = nx.spring_layout(G) # positions for all nodes

	# nodes
	nx.draw_networkx_nodes(G, pos, node_size = 200)

	# edges
	nx.draw_networkx_edges(G, pos, edgelist = elarge, width = 0.4)
	nx.draw_networkx_edges(G, pos, edgelist = esmall, width = 0.4, alpha = 0.6, style = 'dashed')

	# labels
	nx.draw_networkx_labels(G, pos, font_size = 6, font_family = 'sans-serif')

	print 'number of cliques/clusters:', nx.graph_number_of_cliques(G)
	print 'time:', time.time() - start
	plt.show()

	#filename = "w_graph" + ".png"
	#plt.savefig(filename)

def constructive_phase(weights, G, N, M):
	for k in range(0, M-1):
		S = dict()
		if N < 15:
			# let L be a list with every edge of the graph
			L = weights
		else:
			n_h = min(0.1 * N * (N-1)/2, len(G.edges()))
			# L = list with the n_h highest weighted edges
			L = nlargest(int(n_h), weights)

		cmin = min(L)
		cmax = max(L)

		alpha = random.random()
		RCL = list()

		for e in G.edges():
			if G[e[0]][e[1]]['weight'] >= cmax + alpha*(cmin-cmax) and G[e[0]][e[1]]['weight'] <= cmax:
				RCL.append(e)

		# choose randomly an element from RCL. This edge connects a node i to another node j inside the graph.
		r = random.randint(0, len(RCL)-1)
		i = RCL[r][0]
		j = RCL[r][1]
		
		intersec = set(G.neighbors(i)).intersection(G.neighbors(j))
		
		# let S_i and S_j be two empty sets of nodes.
		S[i] = set()
		S[j] = set()
		S[i].add(i)
		S[j].add(j)

		# consider all elements v such that (v, i) in G and (v, j) in G, i != j.
		for v in intersec:
			if G[v][i]['weight'] < G[v][j]['weight']:
				S[i].add(v)
			else:
				S[j].add(v)

		weights = list()

		# eliminate from the graph G the edges (v_t , v_z) in V where v_t in S_i and v_z in S_j
		for e in G.edges():
			if (e[0] in S[i] and e[1] in S[j]) or (e[0] in S[j] and e[1] in S[i]):
				G.remove_edge(e[0], e[1])
			else:
				weights.append(G[e[0]][e[1]]['weight'])

		#for i1 in range(0, len(intersec)):
		#	for i2 in range(i1+1, len(intersec)):
		#		v1 = list(intersec)[i1]
		#		v2 = list(intersec)[i2]
		#		if ((v1 in S[i]) and (v2 in S[j])) or ((v2 in S[i]) and (v1 in S[j])):
		#			G.remove_edge(v1, v2)
		#		else:
		#			weights.append(G[v1][v2]['weight'])

	initial_solution = numpy.zeros(N)
	clusters = list(nx.find_cliques(G))

	for i1 in range(0, len(clusters)):
		for j1 in range(0, len(clusters[i1])):
			initial_solution[clusters[i1][j1]] = i1
	
	#draw_graph(G)
	return initial_solution

def solution_cost(solution, G_full):
	cost = 0
	N = G_full.__len__()
	for i in range(0, N-1):
		for j in range(i+1, N):
			if solution[i] == solution[j]:
				y = 1
			else:
				y = 0
			cost = cost + G_full[i][j]['weight']*y

	return cost

def local_search(solution, G_full, M):
	N = G_full.__len__()

	it = 0
	while it < 5:
		for i in range(0, N):
			sum_i = numpy.zeros(M)
			min_sum = 999999
			c = 0
			for k in range(0, M):
				for j in range(0, N):
					if i != j:
						if solution[j] == k:
							x = 1
						else:
							x = 0
						sum_i[k] = sum_i[k] + G_full[i][j]['weight']*x
				if sum_i[k] < min_sum:
					min_sum = sum_i[k]
					c = k

			solution[i] = c
		
		it = it+1
		print 'f(x%d) = %f' % (it, solution_cost(solution, G_full))
	
	return solution

def demo():
	mydata = load_data('iris.csv')

	N = len(mydata) # number of instances
	M = 3 # number of clusters

	# removing last field: label/classification
	mydata_att = numpy.delete(mydata, 4, 1)

	# converting matrix (originally string) to float
	mydata_att = mydata_att.astype(numpy.float)

	# initializing distance matrix/graph
	G = nx.Graph()
	G_full= nx.Graph()
	weights = list()
	
	# calculating distance matrix: euclidean distance
	for i in range(0, N):
		for j in range(i+1, N):
			w = distance.euclidean(mydata_att[i], mydata_att[j])
			G.add_edge(i, j, weight = w)
			G_full.add_edge(i, j, weight = w)
			weights.append(w)

	initial_solution = constructive_phase(weights, G, N, M)
	print 'f(x%d) = %f' % (0, solution_cost(initial_solution, G_full))
	solution = local_search(initial_solution, G_full, M)

	eval_solution(initial_solution, mydata[:,4], N)

	print 'time:', time.time() - start

def eval_solution(solution, results, N):
	a = 0
	b = 0
	c = 0
	
	# count the number of pairs that are in the same cluster under C and in the same class under C'
	for i in range(0, N):
		for j in range(i+1, N):
			if solution[i] == solution[j] and results[i] == results[j]:
				a = a + 1

	counter_solution = Counter(solution)
	counter_results = Counter(results)

	# clusters
	for k1 in counter_solution:
		b = b + (counter_solution[k1])*(counter_solution[k1] - 1)/2

	# classes
	for k2 in counter_results:
		c = c + (counter_results[k2])*(counter_results[k2] - 1)/2

	b = b - a
	c = c - a
	d = N*(N-1)/2 - a - b - c

	print 'rand:', rand(a, b, c, d)
	print 'crand:', crand(a, b, c, d)

def rand(a, b, c, d):
	total = a+b+c+d
	rand = 1.0*(a+d)/total
	return rand

def crand(a, b, c, d):
	total = a+b+c+d
	crand = (a - (1.0*(b+a)*(c+a))/total)/((1.0*(b+a+c+a))/2 - (1.0*(b+a)*(c+a))/total)
	return crand

if __name__ == "__main__":
	start = time.time()
	demo()