import hnswlib
from collections import deque
import numpy as np

class ANN:
    def __init__(self, d, space='l2', max_size=1000100, ef=50, M=16, batch_add=-1):
        self.dim = d
        self.space = space
        self.max_size = max_size
        self.ef = ef
        self.M = M
        self.index = hnswlib.Index(space = self.space, dim = self.dim)
        self.index.init_index(max_elements = int(1.2*self.max_size), ef_construction = self.ef, M = self.M)

        self.data = deque(maxlen=self.max_size)
        self.queue = []
        self.n_added = 0
        self.batch_add = batch_add

    def add(self, point):
        if self.batch_add > 0 and len(self.queue) > self.batch_add:
            self.data.extend(self.queue)
            self.index.add_items(self.queue)
            self.queue = []
        elif self.batch_add > 0:
            self.queue.append(point)
        else:
            # self.data.append(point)
            self.index.add_items(point)
        # if self.index.get_current_count() > self.data.maxlen*1.1:
            # self.refresh()

    def nearest(self, points):
        labels, distances = self.index.knn_query(points, k=1)
        obs = np.array(self.index.get_items(labels))
        return labels, distances, obs

    def refresh(self):
        new_index = hnswlib.Index(space = self.space, dim = self.dim)
        new_index.init_index(max_elements = int(1.2*self.max_size), ef_construction = self.ef, M = self.M)
        new_index.add_items(self.data)
        self.index = new_index

# import time
# start = time.time()
# last_time = start
# dim = 30
# n_test = 2000
# ann = ANN(d=dim, max_size=100000)
# for i in range(1000000):
#     test_points = np.float32(np.random.random((n_test, dim)))
#     if i > 1000:
#         labels, distances = ann.nearest(test_points)
#     ann.add(test_points[0])
#     if i%1000 == 999:
#         print('Time to '+str(i+1)+': '+str(round(time.time()-last_time, 3)))
#         last_time = time.time()
