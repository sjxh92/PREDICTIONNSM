import numpy as np
import matplotlib.pyplot as plt
import TrafficModel
from TrafficModel import Request
import MetroNetwork
from MetroNetwork import NetworkEnvironment
import networkx as nx
import constants as C
import logging

logger = logging.getLogger(__name__)
handler = logging.FileHandler("results1.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

GAMMA_HIGH = 1
GAMMA_LOW = 0


def trafficGenerate(n, network, total_time):
    traffic_model = TrafficModel.SliceGenerator(n, network, total_time)
    slices = traffic_model.randomGenerate()
    return slices


class PredictionNSM(object):
    def __init__(self, network: NetworkEnvironment, total_time: int, slices: list):
        self.network = network
        self.slices = slices
        self.time = 0
        self.total_time = total_time
        self.transfer_traffic = np.zeros(shape=(total_time,), dtype=np.int)
        self.failure = np.zeros(shape=(total_time,), dtype=np.int)
        self.penalty = 0

    def initialize(self, reservation: float):
        for s in self.slices:
            src = s.src
            dst = s.dst
            paths = self.ksp(src, dst, 3)
            is_avai = False
            ranked_paths = self.network.path_rank(paths=paths)
            if s.priority == 1:
                traffic = int(s.traffic[0] * (1 + reservation))
            else:
                traffic = s.traffic[0]

            traffic_list = self.trafficProcess(traffic)

            path_state = []
            node_state = []
            wave_state = []
            success = False

            for path_index in ranked_paths:
                path = paths[path_index]
                path_state, node_state, wave_state = self.network.set_traffic_list_to_path(s.index, traffic_list, path)
                if path_state:
                    s.add_allocation(0, path_state, node_state, wave_state, True, s.traffic[0], traffic)
                    success = True
                    break
            if not success:
                self.failure[0] += 1
                s.add_allocation(0, -1, -1, -1, False, s.traffic[0], 0)
            # print edge state
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^sss')
            # path = s.state.loc[0]['path']
            # if path != -1:
            #     edges = self.network.extract_path(path)
            # print(path)
            # for edge in edges:
            #     print(edge, " ", self.network.get_edge_data(edge[0], edge[1])['capacity'])
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^eee')

    def Fit(self, time: int):
        ranked_slice = self.slicePreProcess(time=time)
        for i in ranked_slice:
            slice_i = self.slices[i]
            #print('########################slice id: ', slice_i.index)
            traffic_t = slice_i.traffic[time]
            balance_t = slice_i.traffic[time] - slice_i.state.loc[time - 1]['allocation']
            path_t_1 = slice_i.state.loc[time - 1]['path']
            if path_t_1 == -1:
                path_t_1 = self.ksp(slice_i.src, slice_i.dst, 1)[0]
            traffic_list = self.trafficProcess(traffic_t)
            is_avai = False
            if balance_t > 0:
                # increasing traffic
                edges = self.network.extract_path(path_t_1)
                self.resourceRelease(slice_i, time - 1)
                available = True
                if_mapped = False
                for edge in edges:
                    # judge if enough
                    if not self.network.is_allocable_edge(edge, traffic_list):
                        available = False
                        break
                if available:
                    # increasing traffic and enough
                    path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                               traffic_list,
                                                                                               path_t_1)
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
                    if_mapped = True
                else:
                    self.resourceRecovery(slice_i, time - 1)
                    self.failure[time] += 1
                    if slice_i.priority == 1:
                        self.penalty += C.HIGH_PRICE * balance_t
                    else:
                        self.penalty += C.LOW_PRICE * balance_t

                    path_state = slice_i.state.loc[time - 1]['path']
                    node_state = slice_i.state.loc[time - 1]['node']
                    wave_state = slice_i.state.loc[time - 1]['wave']
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                           traffic_t, slice_i.state.loc[time - 1]['allocation'])
            elif balance_t < 0:
                # decreasing traffic
                self.resourceRelease(slice_i, time - 1)
                path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index, traffic_list,
                                                                                           path_t_1)
                if not path_state:
                    slice_i.add_allocation(time, -1, -1, -1, False, traffic_t, 0)
                slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
            else:
                # equal
                path_state = slice_i.state.loc[time - 1]['path']
                node_state = slice_i.state.loc[time - 1]['node']
                wave_state = slice_i.state.loc[time - 1]['wave']
                slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)

    def OverProvision(self, time: int, reservation: float):

        ranked_slice = self.slicePreProcess(time=time)
        for i in ranked_slice:
            slice_i = self.slices[i]
            # print('########################slice id: ', slice_i.index)
            traffic_t = slice_i.traffic[time]
            balance_t = slice_i.traffic[time] - slice_i.state.loc[time - 1]['allocation']
            path_t_1 = slice_i.state.loc[time - 1]['path']
            if path_t_1 == -1:
                path_t_1 = self.ksp(slice_i.src, slice_i.dst, 1)[0]

            if balance_t > 0:
                edges = self.network.extract_path(path_t_1)
                self.resourceRelease(slice_i, time - 1)
                available = 2
                if slice_i.priority == 0:
                    traffic_list = self.trafficProcess(traffic_t)
                    if_mapped = False
                    for edge in edges:
                        # judge if enough
                        if not self.network.is_allocable_edge(edge, traffic_list):
                            available = 0
                            break
                else:
                    traffic_list = self.trafficProcess(int(traffic_t * (1 + reservation)))
                    for edge in edges:
                        # judge if enough
                        if not self.network.is_allocable_edge(edge, traffic_list):
                            available = 1
                            break
                    if available == 1:
                        traffic_list = self.trafficProcess(traffic_t)
                        for edge in edges:
                            # judge if enough
                            if time == 14:
                                print(self.network.get_edge_data(edge[0], edge[1])['capacity'], traffic_list)
                            if not self.network.is_allocable_edge(edge, traffic_list):
                                available = 0
                                break

                if available > 0:
                    # increasing traffic and enough
                    path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                               traffic_list,
                                                                                               path_t_1)
                    if not path_state:
                        print(slice_i.priority)
                        print(traffic_t)
                        print(traffic_list)
                        print(available)
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t,
                                           np.sum(traffic_list))
                else:
                    self.resourceRecovery(slice_i, time - 1)
                    self.failure[time] += 1
                    if slice_i.priority == 1:
                        self.penalty += C.HIGH_PRICE * balance_t
                    else:
                        self.penalty += C.LOW_PRICE * balance_t

                    path_state = slice_i.state.loc[time - 1]['path']
                    node_state = slice_i.state.loc[time - 1]['node']
                    wave_state = slice_i.state.loc[time - 1]['wave']
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                           traffic_t, slice_i.state.loc[time - 1]['allocation'])
            elif balance_t < 0:
                if slice_i.traffic[time] > slice_i.traffic[time - 1]:
                    self.resourceRelease(slice_i, time - 1)
                    edges = self.network.extract_path(path_t_1)
                    available = 1
                    if slice_i.priority == 0:
                        traffic_list = self.trafficProcess(traffic_t)
                        if_mapped = False
                        for edge in edges:
                            # judge if enough
                            if not self.network.is_allocable_edge(edge, traffic_list):
                                available = 0
                                break
                    else:
                        traffic_list = self.trafficProcess(int(traffic_t * (1 + reservation)))
                        if_mapped = False
                        for edge in edges:
                            # judge if enough
                            if not self.network.is_allocable_edge(edge, traffic_list):
                                available = 0
                                break
                    if available == 1:
                        # scale up and enough
                        path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                                   traffic_list,
                                                                                                   path_t_1)
                        if not path_state:
                            assert 0
                        if slice_i.priority == 0:
                            slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
                        else:
                            slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t,
                                                   int(traffic_t * (1 + reservation)))
                    else:
                        self.resourceRecovery(slice_i, time - 1)
                        path_state = slice_i.state.loc[time - 1]['path']
                        node_state = slice_i.state.loc[time - 1]['node']
                        wave_state = slice_i.state.loc[time - 1]['wave']
                        if not path_state:
                            assert 0
                        slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                               traffic_t, slice_i.state.loc[time - 1]['allocation'])
                elif slice_i.traffic[time] < slice_i.traffic[time - 1]:
                    # decreasing traffic
                    self.resourceRelease(slice_i, time - 1)
                    edges = self.network.extract_path(path_t_1)
                    available = 1
                    if slice_i.priority == 0:
                        traffic_list = self.trafficProcess(traffic_t)
                        for edge in edges:
                            # judge if enough
                            if not self.network.is_allocable_edge(edge, traffic_list):
                                available = 0
                                break
                    else:
                        traffic_list = self.trafficProcess(int(traffic_t * (1 + reservation)))
                        for edge in edges:
                            # judge if enough
                            if not self.network.is_allocable_edge(edge, traffic_list):
                                available = 0
                                break
                    if available == 1:
                        path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                                   traffic_list,
                                                                                                   path_t_1)
                        if not path_state:
                            assert 0
                        if slice_i.priority == 0:
                            slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
                        else:
                            slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t,
                                                   int(traffic_t * (1 + reservation)))
                    else:
                        self.resourceRecovery(slice_i, time - 1)
                        path_state = slice_i.state.loc[time - 1]['path']
                        node_state = slice_i.state.loc[time - 1]['node']
                        wave_state = slice_i.state.loc[time - 1]['wave']
                        slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                               traffic_t, slice_i.state.loc[time - 1]['allocation'])
                        if not path_state:
                            print('traffic in t-2', slice_i.traffic[time - 2])
                            print('release traffic', slice_i.state.loc[time - 2]['allocation'])
                            print('traffic in t-1', slice_i.traffic[time - 1])
                            print('release traffic', slice_i.state.loc[time - 1]['allocation'])
                            print('traffic list', traffic_list)
                            print('traffic ', traffic_t)
                            print('priority: ', slice_i.priority)
                            assert 0
                else:
                    # equal
                    path_state = slice_i.state.loc[time - 1]['path']
                    node_state = slice_i.state.loc[time - 1]['node']
                    wave_state = slice_i.state.loc[time - 1]['wave']
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                           traffic_t, slice_i.state.loc[time - 1]['allocation'])
            else:
                self.resourceRelease(slice_i, time - 1)
                edges = self.network.extract_path(path_t_1)
                available = 1
                if slice_i.priority == 0:
                    traffic_list = self.trafficProcess(traffic_t)
                    if_mapped = False
                    for edge in edges:
                        # judge if enough
                        if not self.network.is_allocable_edge(edge, traffic_list):
                            available = 0
                            break
                else:
                    traffic_list = self.trafficProcess(int(traffic_t * (1 + reservation)))
                    if_mapped = False
                    for edge in edges:
                        # judge if enough
                        if not self.network.is_allocable_edge(edge, traffic_list):
                            available = 0
                            break
                if available == 1:
                    path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                               traffic_list,
                                                                                               path_t_1)
                    if not path_state:
                        assert 0
                    if slice_i.priority == 0:
                        slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
                    else:
                        slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t,
                                               int(traffic_t * (1 + reservation)))
                else:
                    self.resourceRecovery(slice_i, time - 1)
                    path_state = slice_i.state.loc[time - 1]['path']
                    node_state = slice_i.state.loc[time - 1]['node']
                    wave_state = slice_i.state.loc[time - 1]['wave']
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                           traffic_t, slice_i.state.loc[time - 1]['allocation'])

    def adjustPrediction(self, time, prediction: bool):
        ranked_slice = self.slicePreProcess(time=time)
        print(ranked_slice)
        for i in ranked_slice:
            slice_i = self.slices[i]
            #print('########################slice id: ', slice_i.index)
            traffic_t = slice_i.traffic[time]
            balance_t = slice_i.traffic[time] - slice_i.state.loc[time - 1]['allocation']
            path_t_1 = slice_i.state.loc[time - 1]['path']
            if path_t_1 == -1:
                path_t_1 = self.ksp(slice_i.src, slice_i.dst, 1)[0]
            traffic_list = self.trafficProcess(traffic_t)

            is_avai = False
            if balance_t > 0:
                edges = self.network.extract_path(path_t_1)
                self.resourceRelease(slice_i, time - 1)
                available = True
                if_mapped = False
                for edge in edges:
                    if not self.network.is_allocable_edge(edge, traffic_list):
                        available = False
                        break
                if available:
                    # increasing traffic and enough
                    path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                               traffic_list,
                                                                                               path_t_1)
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
                    if_mapped = True
                else:
                    # increasing traffic and not enough
                    src = slice_i.src
                    dst = slice_i.dst
                    paths = self.ksp(src, dst, 5)
                    if prediction:
                        ranked_paths = self.predictedPath(paths, time, C.N_STEPS)
                    else:
                        ranked_paths = self.network.path_rank(paths)
                    for path_index in ranked_paths:
                        path = paths[path_index]
                        path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index,
                                                                                                   traffic_list,
                                                                                                   path)
                        if path_state:
                            slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
                            self.transfer_traffic[time] = traffic_t
                            if_mapped = True
                            break
                if not if_mapped:
                    # not enough and no success
                    self.resourceRecovery(slice_i, time - 1)
                    self.failure[time] += 1
                    if slice_i.priority == 1:
                        self.penalty += C.HIGH_PRICE * balance_t
                    else:
                        self.penalty += C.LOW_PRICE * balance_t

                    path_state = slice_i.state.loc[time - 1]['path']
                    node_state = slice_i.state.loc[time - 1]['node']
                    wave_state = slice_i.state.loc[time - 1]['wave']
                    if not path_state:
                        assert 0
                    slice_i.add_allocation(time, path_state, node_state, wave_state, False,
                                           traffic_t, slice_i.state.loc[time - 1]['allocation'])
            elif balance_t < 0:
                # decreasing traffic
                self.resourceRelease(slice_i, time - 1)
                path_state, node_state, wave_state = self.network.set_traffic_list_to_path(slice_i.index, traffic_list,
                                                                                           path_t_1)
                if not path_state:
                    slice_i.add_allocation(time, -1, -1, -1, False, traffic_t, 0)
                slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)
            else:
                path_state = slice_i.state.loc[time - 1]['path']
                node_state = slice_i.state.loc[time - 1]['node']
                wave_state = slice_i.state.loc[time - 1]['wave']
                slice_i.add_allocation(time, path_state, node_state, wave_state, True, traffic_t, traffic_t)

            # # print edge state
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^sss, time: ', time, 'slice id: ', slice_i.index, 'traffic',
            #       slice_i.traffic[time])
            # path = slice_i.state.loc[time]['path']
            # print(path)
            # edges = self.network.extract_path(path)
            # for edge in edges:
            #     print(edge, " ", self.network.get_edge_data(edge[0], edge[1])['capacity'])
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^eee')

    def resourceRelease(self, s: Request, time: int):
        path = s.state.loc[time]['path']
        if path == -1:
            return
        node = s.state.loc[time]['node']
        wave = s.state.loc[time]['wave']
        traffic = s.state.loc[time]['allocation']
        traffic_list = self.trafficProcess(traffic)
        edges = self.network.extract_path(path)
        for i in range(len(edges)):
            edge = edges[i]
            waves = wave[i]
            t = 0
            # print('slice 6: ', s.state['path'])
            # print('time', time)
            # print('traffic at t', s.traffic[time])
            # print('resource_release', edge, self.network.get_edge_data(edge[0], edge[1])['capacity'])
            # print('resource_release', waves)
            # print('resource_release', traffic_list)
            for w in waves:
                self.network.set_wave_capacity_edge(edge, traffic_list[t], w, s.index, False, False)
                t += 1
                # self.network.get_edge_data(edge[0], edge[1])['capacity'][time][w] += traffic_list[w]
        node = node[0]
        self.network.set_node_state(node, traffic, False, False)

    def resourceRecovery(self, s: Request, time: int):
        path = s.state.loc[time]['path']
        if path == -1:
            return
        node = s.state.loc[time]['node']
        wave = s.state.loc[time]['wave']
        traffic = s.state.loc[time]['allocation']
        traffic_list = self.trafficProcess(traffic)
        edges = self.network.extract_path(path)
        for i in range(len(edges)):
            edge = edges[i]
            waves = wave[i]
            t = 0
            for w in waves:
                self.network.set_wave_capacity_edge(edge, traffic_list[t], w, s.index, True, False)
                t += 1
                # self.network.get_edge_data(edge[0], edge[1])['capacity'][time][w] += traffic_list[w]
        node = node[0]
        self.network.set_node_state(node, traffic, False, False)

    def trafficProcess(self, traffic: int):
        a = traffic // 10
        b = traffic - (a * 10)
        traffic_list = []
        if b > 0:
            traffic_list.append(b)
        for i in range(a):
            traffic_list.append(10)
        return traffic_list

    def slicePreProcess(self, time):

        balance_traffic_1 = []
        balance_traffic_0 = []
        down_slice = []
        up_slice = []
        up_slice_1 = []
        up_slice_0 = []
        ranked_up_slice_1 = []
        ranked_up_slice_0 = []
        ranked_slice = []
        for s in self.slices:
            balance = s.traffic[time] - s.traffic[time - 1]
            if balance <= 0:
                down_slice.append(s.index)
            else:
                up_slice.append(s.index)

        for t in up_slice:
            if self.slices[t].priority == 1:
                up_slice_1.append(t)
            else:
                up_slice_0.append(t)

        ##for the high priority
        for i in up_slice_1:
            balance = self.slices[i].traffic[time] - self.slices[i].traffic[time - 1]
            balance_traffic_1.append(balance)
        balance_traffic_1 = np.array(balance_traffic_1)
        balance_traffic_1 = np.argsort(-balance_traffic_1)
        ##for the low prioirty
        for j in up_slice_0:
            balance = self.slices[j].traffic[time] - self.slices[j].traffic[time - 1]
            balance_traffic_0.append(balance)
        balance_traffic_0 = np.array(balance_traffic_0)
        balance_traffic_0 = np.argsort(-balance_traffic_0)

        for m in balance_traffic_1:
            ranked_up_slice_1.append(up_slice_1[m])
        print(ranked_up_slice_1)
        for n in balance_traffic_0:
            ranked_up_slice_0.append(up_slice_0[n])
        print(ranked_up_slice_0)
        down_slice.extend(ranked_up_slice_1)
        down_slice.extend(ranked_up_slice_0)
        return down_slice

    def predictedPath(self, candidate_paths, time, n_step):

        paths_residual_bandwidth = []
        for path in candidate_paths:
            path_residual_bandwidth = 0
            edges = self.network.extract_path(path)
            for edge in edges:
                slices_index = self.network.statistics_edge(edge)
                high_slices_up = []
                high_slices_down = []
                low_slice_up = []
                low_slices_down = []
                edge_bandwidth = 0
                for index in slices_index:
                    s = self.slices[index]
                    if s.priority == 1 and s.traffic[time] > s.traffic[time - 1]:
                        high_slices_up.append(index)
                    elif s.priority == 1 and s.traffic[time] < s.traffic[time - 1]:
                        high_slices_down.append(index)
                    elif s.priority == 0 and s.traffic[time] > s.traffic[time - 1]:
                        low_slice_up.append(index)
                    elif s.priority == 0 and s.traffic[time] < s.traffic[time - 1]:
                        low_slices_down.append(index)
                    else:
                        pass
                for s1 in high_slices_up:
                    edge_bandwidth += self.formula_1(s1, time, n_step, GAMMA_HIGH)
                for s2 in high_slices_down:
                    edge_bandwidth -= self.slices[s2].traffic[time] - self.slices[s2].traffic[time - 1]
                for s3 in low_slices_down:
                    edge_bandwidth -= self.slices[s3].traffic[time] - self.slices[s3].traffic[time - 1]
                path_residual_bandwidth += C.WAVE_NUM * C.WAVE_CAPACITY - edge_bandwidth
            paths_residual_bandwidth.append(path_residual_bandwidth)
        paths_residual_bandwidth = np.array(paths_residual_bandwidth)
        rank = np.argsort(-paths_residual_bandwidth)
        return rank

    def formula_1(self, i, t, n, gamma):
        current_traffic = self.slices[i].traffic[t]
        numpy_traffic = np.array(self.slices[i].traffic)
        n_traffic = numpy_traffic[t: t+n: 1]
        max_traffic = np.amax(n_traffic)
        allocated_bandwidth = current_traffic + gamma * max_traffic
        return allocated_bandwidth

    def resetSlices(self):
        for s in self.slices:
            times = np.arange(C.TOTAL_TIME)
            if not s.state.empty:
                s.state.drop(index=times, axis=0, inplace=True)

    def capacityPrediction(self, time):
        pass

    def findNode(self, time):
        pass

    def findPath(self, time):
        pass

    def ksp(self, source, target, k):
        """
        calculate the paths
        :param k:
        :param source:
        :param target:
        :return:
        """
        if source is None:
            return [None]
        paths = nx.shortest_simple_paths(self.network, source, target)
        path_k = []
        index = 0
        for i in paths:
            index += 1
            if index > k:
                break
            path_k.append(i)
        return path_k

    def showRequest(self):
        for s in self.slices:
            print('|slice id: ', s.index, '|slice src: ', s.src,
                  '|slice dst: ', s.dst, '|slice traffic', s.traffic,
                  '|slice priority: ', s.priority, '|total time: ', s.t_time)
            if not s.state.empty:
                print(s.state)

    def showResults(self):
        pass

    def drawRequest(self):
        x = np.arange(self.total_time)
        y = np.zeros(shape=(len(self.slices), self.total_time), dtype=np.int)
        for i in range(len(self.slices)):
            y[i] = self.slices[i].traffic
            plt.plot(x, y[i], label=self.slices[i].src)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('traffic')
        plt.show()

    def test(self):
        edge = [5, 2]
        traffic_list = [3, 10, 10, 10]
        avai = self.network.is_allocable_edge(edge, traffic_list)
        print(avai)

    def showPath(self, path: list):
        edges = self.network.extract_path(path)
        for edge in edges:
            for w in range(self.network.wave_num):
                print(self.network.get_edge_data(edge[0], edge[1])['capacity'][w])

    def output_to_logfile(self, prediction: str, n: int):
        logger.info('It is the ' + str(n) + " simulation times, and the mode is: " + prediction)
        logger.info('the penalty is: ' + str(self.penalty))
        # logger.info(self.transfer_traffic)
        logger.info('the transfer traffic is: ' + str(np.sum(self.transfer_traffic)))
        # logger.info(self.failure)
        logger.info('++++++++++++++++++++++++++++++++')


def main(network: MetroNetwork.NetworkEnvironment, n: int, prediction: bool, mode: int,
         reservation: float, slices: list):
    network.reset()
    heuristic = PredictionNSM(network, C.TOTAL_TIME, slices)
    heuristic.resetSlices()
    heuristic.initialize(reservation)
    for i in range(1, C.TOTAL_TIME):
        if mode == 1:
            heuristic.Fit(i)
        elif mode == 2:
            heuristic.OverProvision(i, reservation)
        else:
            heuristic.adjustPrediction(i, prediction)
        # heuristic.showRequest()
    heuristic.output_to_logfile(str(prediction), n)
    heuristic.network.usage_edge()
    print(np.sum(heuristic.failure))
    return heuristic.penalty, np.sum(heuristic.transfer_traffic), heuristic.failure


if __name__ == "__main__":

    penalty_list1 = []
    transfer_list1 = []
    penalty_list2 = []
    transfer_list2 = []
    penalty_list3 = []
    transfer_list3 = []
    penalty_list4 = []
    transfer_list4 = []
    failure1 = []
    failure2 = []
    failure3 = []
    failure4 = []

    slice_num = 13

    network = MetroNetwork.NetworkEnvironment("LargeTopology_link",
                                              "/home/mario/PycharmProjects/PredictionNSM/Resource")
    # slices = trafficGenerate(1, network, C.TOTAL_TIME)
    # heuristic = PredictionNSM(network, C.TOTAL_TIME, slices)
    # heuristic.showRequest()

    iteration = 1
    for i in range(0, slice_num):
        penalty = np.zeros(shape=(4, iteration))
        transfer = np.zeros(shape=(4, iteration))

        for j in range(iteration):
            slices = trafficGenerate(i, network, C.TOTAL_TIME)
            penalty[0][j], transfer[0][j], failure1 = main(network, n=i, prediction=False, mode=1, reservation=0, slices=slices)
            penalty[1][j], transfer[1][j], failure2 = main(network, n=i, prediction=False, mode=2, reservation=0.3, slices=slices)
            penalty[2][j], transfer[2][j], failure3 = main(network, n=i, prediction=False, mode=3, reservation=0, slices=slices)
            penalty[3][j], transfer[3][j], failure4 = main(network, n=i, prediction=True, mode=3, reservation=0, slices=slices)
        penalty_list1.append(np.mean(penalty[0]))
        penalty_list2.append(np.mean(penalty[1]))
        penalty_list3.append(np.mean(penalty[2]))
        penalty_list4.append(np.mean(penalty[3]))
        transfer_list1.append(np.mean(transfer[0]))
        transfer_list2.append(np.mean(transfer[1]))
        transfer_list3.append(np.mean(transfer[2]))
        transfer_list4.append(np.mean(transfer[3]))
    file = open("/home/mario/PycharmProjects/PredictionNSM/Results/penalty", "a+")
    file.write('the penalty for fit strategy\n')
    file.write(str(penalty_list1) + "\n")
    file.write('the penalty for overprovision strategy\n')
    file.write(str(penalty_list2) + "\n")
    file.write('the penalty for adjust without prediction strategy\n')
    file.write(str(penalty_list3) + "\n")
    file.write('the penalty for adjust with prediction strategy\n')
    file.write(str(penalty_list4) + "\n")

    file.write('the transfer traffic for fit strategy\n')
    file.write(str(transfer_list1) + "\n")
    file.write('the transfer traffic for overprovision strategy\n')
    file.write(str(transfer_list2) + "\n")
    file.write('the transfer traffic for adjust without prediction strategy\n')
    file.write(str(transfer_list3) + "\n")
    file.write('the transfer traffic for adjust with prediction strategy\n')
    file.write(str(transfer_list4) + "\n")

    file.write('the failure for fit strategy\n')
    file.write(str(failure1) + "\n")
    file.write('the failure for overprovision strategy\n')
    file.write(str(failure2) + "\n")
    file.write('the failure for adjust without prediction strategy\n')
    file.write(str(failure3) + "\n")
    file.write('the failure for adjust with prediction strategy\n')
    file.write(str(failure4) + "\n")
    file.close()

    x = np.arange(slice_num)
    plt.figure()
    plt.plot(x, penalty_list1, c='red', marker='s', ms=4, label='Fit')
    plt.plot(x, penalty_list2, c='green', marker='o', ms=4, label='Over Provision')
    plt.plot(x, penalty_list3, c='blue', marker='s', ms=4, label='With PREDICTION')
    plt.plot(x, penalty_list4, c='orange', marker='o', ms=4, label='PREDICTION')
    plt.legend()
    plt.xlabel('slices number')
    plt.ylabel('penalty')
    #
    plt.figure()
    plt.plot(x, transfer_list1, c='red', marker='s', ms=4, label='Fit')
    plt.plot(x, transfer_list2, c='green', marker='o', ms=4, label='Over Provision')
    plt.plot(x, transfer_list3, c='blue', marker='s', ms=4, label='With PREDICTION')
    plt.plot(x, transfer_list4, c='orange', marker='o', ms=4, label='PREDICTION')
    plt.legend()
    plt.xlabel('slices number')
    plt.ylabel('transfer traffic')

    x = np.arange(C.TOTAL_TIME)
    plt.figure()
    plt.plot(x, failure1, c='red', marker='s', ms=4, label='Fit')
    plt.plot(x, failure2, c='green', marker='o', ms=4, label='Over Provision')
    plt.plot(x, failure3, c='blue', marker='s', ms=4, label='With PREDICTION')
    plt.plot(x, failure4, c='orange', marker='o', ms=4, label='PREDICTION')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('failure')

    plt.show()

    # heuristic.drawRequest()
