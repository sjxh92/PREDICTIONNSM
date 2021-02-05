from args import args
from Game import NSMGame
import numpy as np
from NSMGame import Game as game
from MetroNetwork import NetworkEnvironment as mNetwork
import networkx as nx

"""
use for the comparison with network slice heuristic from Reza's paper
"""


class FFKSP(object):
    def __init__(self, network: mNetwork, k: int):
        self.network = network
        self.success_req = 0
        self.failed_req = []
        self.k = k

    def heuristic(self,
                  req_list,
                  event_list,
                  time_event):
        done = False
        time = 0
        while not done:
            print('++++++++++++', time)
            event_index_list = time_event[time]
            if event_index_list:
                for event_index in event_index_list:
                    event = event_list[event_index]
                    if event[2]:
                        req = req_list[event[1]]
                        src = req.src
                        dst = req.dst
                        path_list = self.ksp(src, dst)
                        if_mapped = False
                        for path in path_list:
                            if self.network.is_allocable(req=req,
                                                         path=path,
                                                         wave_index=0,
                                                         node_index=0,
                                                         demand=req.traffic,
                                                         start_time=req.arrival_time,
                                                         end_time=req.leave_time):
                                physical_node = self.network.set_node_state(start_time=req.arrival_time,
                                                                            end_time=req.leave_time,
                                                                            path=path,
                                                                            node_index=0,
                                                                            demand=req.traffic,
                                                                            state=1)
                                self.network.set_wave_state(start_time=req.arrival_time,
                                                            end_time=req.leave_time,
                                                            wave_index=0,
                                                            path=path,
                                                            state=False,
                                                            check=True)
                                req.add_allocation(path, 0, physical_node)
                                self.success_req += 1
                                if_mapped = True
                                break
                        if not if_mapped:
                            self.failed_req.append(req.index)
                        if req.index == len(req_list) - 1:
                            done = True
                    else:
                        req = req_list[event[1]]
                        start_time = req.arrival_time
                        end_time = req.leave_time
                        traffic = req.traffic
                        if hasattr(req, 'path') and hasattr(req, 'wave') and hasattr(req, 'node'):
                            path = req.path
                            wave = req.wave_index
                            node = req.node_index
                            self.network.set_wave_state(start_time=start_time,
                                                        end_time=end_time,
                                                        wave_index=wave,
                                                        path=path,
                                                        state=True,
                                                        check=True)
                            self.network.set_node_state(start_time=start_time,
                                                        end_time=end_time,
                                                        path=path,
                                                        node_index=node,
                                                        demand=traffic,
                                                        state=0)
            time += 1

    def show_results(self, request):
        print('**************print results**********************')
        for i in request:
            print('request index:', request[i].index)
            print('request src:', request[i].src)
            print('request dst:', request[i].dst)
            print('request start:', request[i].arrival_time)
            print('request end:', request[i].leave_time)
            print('request traffic:', request[i].traffic)
            if hasattr(request[i], 'path'):
                print('request path:', request[i].path)
            if hasattr(request[i], 'wave_index'):
                print('request wave index:', request[i].wave_index)
            if hasattr(request[i], 'node_index'):
                print('request node index:', request[i].node_index)
            print('===========================')
        print('the success request is: ', self.success_req)
        print("the failed requests are: ", self.failed_req)

    def ksp(self, source, target):
        """
        calculate the paths
        :param source:
        :param target:
        :return:
        """
        if source is None:
            return [None]
        paths = nx.shortest_simple_paths(self.network, source, target, weight='weight')
        path_k = []
        index = 0
        for i in paths:
            index += 1
            if index > self.k:
                break
            path_k.append(i)
        return path_k
