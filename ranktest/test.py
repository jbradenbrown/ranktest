

import gen_data
import numpy as np
import model
import pandas

itol = lambda l: [l]

data = gen_data.gen_data(gen_data.get_dists(100), 100000, 10, .2, 1/5, .1)
train = (data[0][:70000], np.array(list(map(itol, data[1][:70000]))))
test = (data[0][70000:], np.array(list(map(itol, data[1][70000:]))))

def train_model():
    x = model.train(list(zip(*train)), list(zip(*test)), list(zip(*test)), 1024, [5,5], epochs = 50)
    return x

def get_res(x):
    res = np.array(list(map(lambda l: [l], x['results'][0][0])))
    return res

def get_pandas():
    return pandas.DataFrame(train[0]).assign(score=train[1])