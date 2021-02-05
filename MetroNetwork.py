import numpy as np
import math
import pandas as pd
import networkx as nx
import os
from itertools import islice
import random
import matplotlib.pyplot as plt
import constants as C

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelno)s - %(name)s - '
                                                '%(levelname)s - %(filename)s - %(funcName)s - '
                                                '%(message)s')
logger = logging.getLogger(__name__)

NODE_NUM = 7
LINK_NUM = 20
NODE_CAPACITY = 500
WAVE_CAPACITY = 10
DESTINATION = 7


class NetworkEnvironment(nx.DiGraph):

    def __init__(self, filename: str, file_prefix: str):
        # self.n_feature = NODE_NUM + 2 + J_NODE * M_VM + K_LINK * W_WAVELENGTH

        # node utilization + link utilization + request node + request traffic + holding time
        super(NetworkEnvironment, self).__init__()
        self.net = None
        self.action_space = []
        self.n_action = len(self.action_space)
        self.memory = np.zeros([0, 4], dtype=int)  # nodeid, traffic, starttime, endtime
        self.sliceId = 0

        filepath = os.path.join(file_prefix, filename)
        if os.path.isfile(filepath):
            datas = np.loadtxt(filepath, delimiter=',', skiprows=0, dtype=str)
            self.origin_data = datas[:, 0:(datas.shape[1])]
        else:
            raise FileExistsError("file {} doesn't exists.".format(filepath))

    def reset(self):
        wave_avai1 = []
        wave_avai2 = []
        self.clear()
        # for i in range(NODE_NUM):
        #     self.add_node(i + 1, capacity=NODE_CAPACITY)  # time step
        for i in range(self.origin_data.shape[0]):
            if self.origin_data[i, 4] == 'Core_link':
                wave_avai1 = 10 * np.ones(shape=(40,), dtype=np.float32)
                wave_avai2 = 10 * np.ones(shape=(40,), dtype=np.float32)
                link_type = 'Core_link'
            elif self.origin_data[i, 4] == 'Extension_link':
                wave_avai1 = 10 * np.ones(shape=(20,), dtype=np.float32)
                wave_avai2 = 10 * np.ones(shape=(20,), dtype=np.float32)
                link_type = 'Extension_link'
            stat1 = np.zeros(shape=(2000,))
            stat2 = np.zeros(shape=(2000,))
            self.add_edge(self.origin_data[i, 2], self.origin_data[i, 3], type=self.origin_data[i, 1],
                          weight=float(self.origin_data[i, 5]), capacity=wave_avai1, stat=stat1)
            self.add_edge(self.origin_data[i, 3], self.origin_data[i, 2], type=self.origin_data[i, 1],
                          weight=float(self.origin_data[i, 5]), capacity=wave_avai2, stat=stat2)


    # *_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*
    def set_wave_capacity_edge(self,
                               edge: list,
                               traffic: int,
                               wave: int,
                               slice_index: int,
                               state: bool,
                               check: bool = True):
        if check:
            print('====')
            print(self.get_edge_data(edge[0], edge[1])['capacity'][wave])
            assert self.get_edge_data(edge[0], edge[1])['capacity'][wave] == 0
        if state:
            balance = self.get_edge_data(edge[0], edge[1])['capacity'][wave]
            if balance < traffic:
                return False
            else:
                self.get_edge_data(edge[0], edge[1])['capacity'][wave] -= traffic
                self.get_edge_data(edge[0], edge[1])['stat'][slice_index] += traffic
                assert self.get_edge_data(edge[0], edge[1])['capacity'][wave] >= 0
                return True
        else:
            self.get_edge_data(edge[0], edge[1])['capacity'][wave] += traffic
            self.get_edge_data(edge[0], edge[1])['stat'][slice_index] -= traffic
            assert self.get_edge_data(edge[0], edge[1])['stat'][slice_index] >= 0
            if self.get_edge_data(edge[0], edge[1])['capacity'][wave] > WAVE_CAPACITY:
                print(self.get_edge_data(edge[0], edge[1])['capacity'][wave])
                assert 0
            return True

    def set_capacity_edge(self,
                          slice_index: int,
                          edge: list,
                          traffic_list: list, ):
        allocated_wave = []
        for t in traffic_list:
            ranked_wave = self.wave_rank_edge(edge=edge)
            for w in ranked_wave:
                balance = self.get_edge_data(edge[0], edge[1])['capacity'][w]
                if balance >= t:
                    self.get_edge_data(edge[0], edge[1])['capacity'][w] -= t
                    self.get_edge_data(edge[0], edge[1])['stat'][slice_index] += np.sum(traffic_list)
                    assert self.get_edge_data(edge[0], edge[1])['capacity'][w] >= 0
                    allocated_wave.append(w)
                    break
        return allocated_wave

    def set_traffic_list_to_path(self,
                                 slice_index: int,
                                 traffic_list: list,
                                 path: list):
        path_state = []
        wave_state = []
        node_state = []
        is_avai = False

        edges = self.extract_path(path)
        edge_available = True
        node_available = True
        for edge in edges:
            if not self.is_allocable_edge(edge, traffic_list):
                edge_available = False
                print('edge capacity is not enough for the path: ', path)
                print('edge capacity is not enough for the edge: ', edge)
                self.showPath(path)
                print(traffic_list)
                break
        for node in path:
            if not self.is_allocable_node(node, np.sum(traffic_list).item()):
                node_available = False
                print('node capacity is not enough')
                break

        if edge_available and node_available:
            is_avai = True
            path_state = path

        if is_avai:
            for edge in edges:
                wave_edge_state = self.set_capacity_edge(slice_index, edge, traffic_list)
                wave_state.append(wave_edge_state)
            for node in path:
                success = self.set_node_state(node_index=node,
                                              traffic=np.sum(traffic_list).item(),
                                              state=True,
                                              check=False)
                if success:
                    node_state.append(node)
                    break
            # print(path_state, node_state, wave_state)
        return path_state, node_state, wave_state

    def set_node_state(self,
                       node_index: int,
                       traffic: int,
                       state: bool,
                       check: bool = True):
        """
        :param check:
        :param node_index:
        :param time:
        :param traffic:
        :param state:
        :return:
        """
        if check:
            assert self.nodes[node_index]['capacity'] == 0
        if state:
            balance = self.nodes[node_index]['capacity']
            if balance < traffic:
                return False
            else:
                self.nodes[node_index]['capacity'] -= traffic
                return True
        else:
            self.nodes[node_index]['capacity'] += traffic
            return True

    def exist_rw_allocation(self, path_list: list, start_time: int, end_time: int) -> [bool, int, int]:
        """
        check all the paths and all the wavelengths in path list
        :param end_time:
        :param start_time:
        :param path_list:
        :return:
        """
        if len(path_list) == 0 or path_list[0] is None:
            return False, -1, -1

        if end_time > self.total_time - 1:
            return False, -1, -1

        for path_index, nodes in enumerate(path_list):
            edges = self.extract_path(nodes)
            for wave_index in range(self.wave_num):
                w_avai = True
                for edge in edges:
                    for time in range(end_time - start_time + 1):
                        if self.get_edge_data(edge[0], edge[1])['capacity'][start_time + time][wave_index] is False:
                            w_avai = False
                            break
                    if w_avai is False:
                        break
                if w_avai is True:
                    return True, path_index, wave_index

        return False, -1, -1

    def wave_rank(self, path: list, time: int):
        edges = self.extract_path(path)
        wave_weight = []
        for wave in range(self.wave_num):
            wave_capacity_edge = 0
            wave_capacity = []
            for edge in edges:
                wave_capacity_edge = self.get_edge_data(edge[0], edge[1])['capacity'][time][wave]
                wave_capacity.append(wave_capacity_edge)
            wave_min = np.amin(wave_capacity, 0)
            wave_weight.append(wave_min)
        wave_weight = np.array(wave_weight)
        sorted_wave_weight = np.argsort(-wave_weight)
        wave_index_sorted = sorted_wave_weight[0:3:1]
        return wave_index_sorted

    def wave_rank_edge(self,
                       edge: list):
        wave_weight = []
        for wave in range(len(self.get_edge_data(edge[0], edge[1])['capacity'])):
            weight = self.get_edge_data(edge[0], edge[1])['capacity'][wave]
            wave_weight.append(weight)
        wave_weight = np.array(wave_weight)
        ranked_wave = np.argsort(wave_weight)
        return ranked_wave

    def node_rank(self,
                  path: list,
                  time: int):
        node_weight = np.arange(len(path))
        physical_node_index = []
        for node in range(len(path)):
            node_capacity = self.nodes[path[node]]['capacity'][time]
            node_weight[node] = node_capacity
        sorted_node_weight = np.argsort(-node_weight)
        sorted_node_weight = sorted_node_weight[0:2:1]
        for node in sorted_node_weight:
            physical_node_index.append(path[node])
        return physical_node_index

    def path_rank(self,
                  paths: list):
        path_weight = []
        for path in paths:
            node_capacity = 0
            for node in path:
                node_capacity += self.nodes[node]['capacity']
            link_capacity = 0
            edges = self.extract_path(path)
            for edge in edges:
                for i in range(len(self.get_edge_data(edge[0], edge[1])['capacity'])):
                    link_capacity += self.get_edge_data(edge[0], edge[1])['capacity'][i]
            path_weight.append(0.5 * node_capacity + 0.5 * link_capacity)
        path_weight = np.array(path_weight)
        ranked_path = np.argsort(-path_weight)
        return ranked_path

    def is_allocable(self,
                     traffic,
                     path_index: list,
                     node_index: int,
                     time: int, ) -> bool:
        """
        if the wave_index in path is available
        :param traffic:
        :param path_index:
        :param time:
        :param node_index:
        :return:
        """
        if time >= self.total_time:
            return False

        edges = self.extract_path(path_index)

        is_avai = True
        for edge in edges:
            # print("the link:", edge[0], "-->", edge[1], "is: ", self.get_edge_data(edge[0], edge[1])['is_wave_avai'])
            path_capacity = np.sum(self.get_edge_data(edge[0], edge[1])['capacity'][time])
            if path_capacity < traffic:
                is_avai = False
                print('\033[1;32;40m the bandwidth is not enough')
                break
        if self.nodes[node_index]['capacity'][time] < traffic:
            is_avai = False
            print('\033[1;32;40m the processing is not enough')
        return is_avai

    def is_allocable_path(self,
                          traffic,
                          path: list,
                          time: int, ) -> bool:
        """
        if the wave_index in path is available
        :param traffic:
        :param path:
        :param time:
        :return:
        """
        if time >= self.total_time:
            return False

        edges = self.extract_path(path)

        is_avai = True
        for edge in edges:
            # print("the link:", edge[0], "-->", edge[1], "is: ", self.get_edge_data(edge[0], edge[1])['is_wave_avai'])
            path_capacity = np.sum(self.get_edge_data(edge[0], edge[1])['capacity'][time])
            if path_capacity < traffic:
                is_avai = False
                print('\033[1;32;40m the bandwidth is not enough')
                break
        return is_avai

    def is_allocable_path_node(self,
                               traffic: list,
                               path: list,
                               time: int, ) -> bool:
        """
        if the wave_index in path is available
        :param traffic:
        :param path:
        :param time:
        :return:
        """
        if time >= self.total_time:
            return False

        edges = self.extract_path(path)

        is_avai = True
        for edge in edges:
            # print("the link:", edge[0], "-->", edge[1], "is: ", self.get_edge_data(edge[0], edge[1])['is_wave_avai'])
            path_capacity = np.sum(self.get_edge_data(edge[0], edge[1])['capacity'][time])
            if path_capacity < traffic:
                is_avai = False
                print('\033[1;32;40m the bandwidth is not enough')
                break
        for node_index in path:
            if self.nodes[node_index]['capacity'][time] < traffic:
                is_avai = False
                print('\033[1;32;40m the processing is not enough')
        return is_avai

    def is_allocable_edge(self,
                          edge: list,
                          traffic: list):
        capacity = []
        for wave in range(len(self.get_edge_data(edge[0], edge[1])['capacity'])):
            capacity.append(self.get_edge_data(edge[0], edge[1])['capacity'][wave])
        # print(capacity)
        capacity.sort()
        #print(capacity)

        success_list = []
        for t in traffic:
            success = False
            for wave in capacity:
                if wave >= t:
                    success = True
                    capacity.remove(wave)
                    break
            success_list.append(success)
        for success in success_list:
            if not success:
                return False
        return True

    def is_allocable_node(self,
                          node: int,
                          traffic: int):
        if self.nodes[node]['capacity'] < traffic:
            return False
        else:
            return True


    def extract_path(self, nodes: list) -> list:
        # print(" extract path ")
        # print(nodes)
        assert len(nodes) >= 2
        rtn = []
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            rtn.append((start_node, end_node))
            start_node = end_node
        return rtn

    def showPath(self, path: list):
        edges = self.extract_path(path)
        for edge in edges:
            print(edge, self.get_edge_data(edge[0], edge[1])['capacity'])

    def statistics_edge(self, edge: list):

        ss = []
        slices = self.get_edge_data(edge[0], edge[1])['stat']
        for i in range(len(slices)):
            if slices[i] > 0:
                ss.append(i)
        return ss

    def usage_edge(self):
        for u, v in self.edges:
            print(self.get_edge_data(u, v)['type'], ':  ', self.get_edge_data(u, v)['capacity'])

    def show_link_state(self):
        for n, nbrs in self.adjacency():
            print(n, nbrs.items())
            for nbr, eattr in nbrs.items():
                data = eattr['weight']
                wave_state = eattr['capacity']
                print('(%d, %d, %d, %0.3f)' % (n, nbr, data, wave_state))


if __name__ == "__main__":
    TP = NetworkEnvironment("LargeTopology_link", "/home/mario/PycharmProjects/PredictionNSM/Resource")
    TP.reset()
    #print(TP.nodes.data())
    print(TP.size())
    TP.clear()
    print(TP.size())
    #nx.draw(TP, with_labels=True)
    #plt.show()

