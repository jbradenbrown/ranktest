

import gen_data
import numpy as np
import model

itol = lambda l: [l]

data = gen_data.gen_data(gen_data.get_dists(100), 100000, 20, .2, 5)
train = (data[0][:70000], np.array(list(map(itol, data[1][:70000]))))
test = (data[0][70000:], np.array(list(map(itol, data[1][70000:]))))

x = model.train(list(zip(*train)), list(zip(*test)), list(zip(*test)), 1024, [5,5], epochs = 200)

res = np.array(list(map(lambda l: [l], x['results'][0][0])))