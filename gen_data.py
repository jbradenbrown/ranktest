# 10 attributes, 3 have different distributions based on resulting score

import random
import numpy as np
import scipy.stats as stats

def get_dists(num):

	dist_vars = lambda: (stats.betaprime.rvs(2,3,0,10), stats.betaprime.rvs(2,3,0,10))

	return [dist_vars() for i in range(num)]

def gen_data(dists, size, pred_num, chance_pred, chg_pred):

	if (pred_num > len(dists)):
		print("Predictive data items must be less than number of distributions")
		sys.exit(1)

	data_nonpred = np.array([norm(stats.norm.rvs(*dist, size)) for dist in dists[:-pred_num]])
	scores = stats.norm.rvs(0,1,size)
	data_pred = np.array([norm(stats.norm.rvs(*dist, size) + ((dist[1] * scores * chg_pred) if random.random() < chance_pred else 0)) for dist in dists[-pred_num:]])
	data = np.append(data_nonpred, data_pred, axis=0).T

	return (data, norm(scores))

def norm(v):
	return (v-min(v))/max(v)