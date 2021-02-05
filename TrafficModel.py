import numpy as np
from MetroNetwork import NetworkEnvironment
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
np.random.seed(0)


class Request(object):
    def __init__(self, index: int, src: int, dst: int, total_time: int, traffic: list, priority: int):
        super(Request, self).__init__()
        self.state = pd.DataFrame(columns=['node', 'path', 'wave', 'success', 'traffic', 'allocation'])
        self.index = index
        self.src = src
        self.dst = dst
        self.t_time = total_time
        self.traffic = traffic
        self.priority = priority

    def add_allocation(self, time: int, path: list, node: int, wave: list, success: bool, traffic: int, allocation: int):
        self.state.loc[time] = [node, path, wave, success, traffic, allocation]


class SliceGenerator(object):
    def __init__(self, n: int, network: NetworkEnvironment, total_time: int):
        self.n = n
        self.network = network
        self.total_time = total_time
        self.slices = []
        pass

    def randomGenerate(self):
        source = []
        for n in self.network.nodes:
            if n.find('AG') >= 0:
                source.append(n)
        print(source)
        slice_index = 0
        for i in range(self.n):
            for n in source:
                src = n
                if np.random.randint(2):
                    dst = 'S_Node_BB_001'
                else:
                    dst = 'S_Node_BB_002'
                traffic = []
                priority = np.random.randint(0, 2)
                if np.random.randint(2):
                    for t in range(self.total_time):
                        t = t % 24
                        if t // 2 == 0: # 0 1
                            traffic_t = np.random.randint(2, 4)
                            traffic.append(traffic_t)
                        elif t // 2 == 1: # 2 3
                            traffic_t = np.random.randint(4, 8)
                            traffic.append(traffic_t)
                        elif t // 2 == 2: # 4 5
                            traffic_t = np.random.randint(4, 12)
                            traffic.append(traffic_t)
                        elif t // 2 == 3: # 6 7
                            traffic_t = np.random.randint(15, 25)
                            traffic.append(traffic_t)
                        elif t // 2 == 4:# 8 9
                            traffic_t = np.random.randint(20, 35)
                            traffic.append(traffic_t)
                        elif t // 2 == 5:# 10 11
                            traffic_t = np.random.randint(30, 45)
                            traffic.append(traffic_t)
                        elif t // 2 == 6: # 12 13
                            traffic_t = np.random.randint(40, 50)
                            traffic.append(traffic_t)
                        elif t // 2 == 7: # 14 15
                            traffic_t = np.random.randint(30, 40)
                            traffic.append(traffic_t)
                        elif t // 2 == 8: # 16 17
                            traffic_t = np.random.randint(15, 30)
                            traffic.append(traffic_t)
                        elif t // 2 == 9: #18 19
                            traffic_t = np.random.randint(15, 35)
                            traffic.append(traffic_t)
                        elif t // 2 == 10: #20 21
                            traffic_t = np.random.randint(10, 30)
                            traffic.append(traffic_t)
                        else: # 22 23
                            traffic_t = np.random.randint(5, 15)
                            traffic.append(traffic_t)
                else:
                    for t in range(self.total_time):
                        t = t % 24
                        if t // 2 == 1: # 0 1
                            traffic_t = np.random.randint(20, 35)
                            traffic.append(traffic_t)
                        elif t // 2 == 2: # 2 3
                            traffic_t = np.random.randint(30, 45)
                            traffic.append(traffic_t)
                        elif t // 2 == 3: # 4 5
                            traffic_t = np.random.randint(40, 50)
                            traffic.append(traffic_t)
                        elif t // 2 == 4: # 6 7
                            traffic_t = np.random.randint(30, 40)
                            traffic.append(traffic_t)
                        elif t // 2 == 5: # 8 9
                            traffic_t = np.random.randint(15, 30)
                            traffic.append(traffic_t)
                        elif t // 2 == 6: # 10 11
                            traffic_t = np.random.randint(15, 35)
                            traffic.append(traffic_t)
                        elif t // 2 == 7: # 12 13
                            traffic_t = np.random.randint(10, 30)
                            traffic.append(traffic_t)
                        elif t // 2 == 8: # 14 15
                            traffic_t = np.random.randint(5, 15)
                            traffic.append(traffic_t)
                        elif t // 2 == 9:  # 16 17
                            traffic_t = np.random.randint(2, 4)
                            traffic.append(traffic_t)
                        elif t // 2 == 10:  # 18 19
                            traffic_t = np.random.randint(4, 8)
                            traffic.append(traffic_t)
                        elif t // 2 == 11:  # 20 21
                            traffic_t = np.random.randint(4, 12)
                            traffic.append(traffic_t)
                        else:  # 22 23
                            traffic_t = np.random.randint(15, 25)
                            traffic.append(traffic_t)
                # print(traffic)
                slice_i = Request(index=slice_index, src=src, dst=dst,
                                  total_time=self.total_time, traffic=traffic, priority=priority)
                self.slices.append(slice_i)
                slice_index += 1
        return self.slices

    def formula_1(self, i, t, n, gamma):
        current_traffic = self.slices[i].traffic[t]
        numpy_traffic = np.array(self.slices[i].traffic)
        n_traffic = numpy_traffic[t: t+n: 1]
        max_traffic = np.amax(n_traffic)
        allocated_bandwidth = current_traffic + gamma * max_traffic
        return allocated_bandwidth

    def showRequest(self):
        for s in self.slices:
            print('|slice id: ', s.index, '|slice src: ', s.src,
                  '|slice dst: ', s.dst, '|slice traffic', s.traffic,
                  '|slice priority: ', s.priority, '|total time: ', s.t_time)
            if not s.state.empty:
                print(s.state)


if __name__ == "__main__":
    TP = NetworkEnvironment("LargeTopology_link", "/home/mario/PycharmProjects/PredictionNSM/Resource")
    TP.reset()
    print(TP.nodes.data())
    print(TP.edges.data())
    generator = SliceGenerator(10, TP, 120)
    generator.randomGenerate()
    generator.showRequest()

