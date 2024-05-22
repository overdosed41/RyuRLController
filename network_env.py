import numpy as np
import time
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink


class NetworkEnv:
    def __init__(self, topo_file, link_states, routing_table):
        self.topo_file = topo_file
        self.link_states = link_states
        self.routing_table = routing_table
        self.net = self.setup_network()
        self.state_dim = len(self.link_states) + len(self.routing_table)
        self.action_dim = len(self.routing_table)
        self.start_time = time.time()

    def setup_network(self):
        # Load network topology from file
        topo = Topo()
        with open(self.topo_file, 'r') as f:
            for line in f:
                node1, node2 = line.strip().split()
                topo.addLink(node1, node2)

        # Create Mininet network
        net = Mininet(topo=topo, link=TCLink)
        net.start()
        return net

    def get_state(self):
        # Get current link states and routing table
        state = np.concatenate((self.link_states, self.routing_table))
        return state

    def get_reward(self, new_state):
        # Calculate reward based on updated network state
        link_states, routing_table = np.split(new_state, [len(self.link_states)])
        # Implement your reward calculation logic here
        reward = 0
        return reward

    def update_state(self, action):
        """
        根据选择的动作更新网络环境的状态。

        参数:
            action (int): 要更新的路由表条目的索引。

        返回:
            np.ndarray: 网络环境的新状态,是当前链路状态和更新后的路由表的拼接。
        """
        # 创建当前路由表的副本,以避免直接修改原有的路由表
        new_routing_table = self.routing_table.copy()

        # 翻转对应于所选动作的路由表条目的值
        # 如果原始值为 1,则变为 0,反之亦然
        new_routing_table[action] = 1 - new_routing_table[action]

        # 将当前链路状态和更新后的路由表拼接成新的状态
        new_state = np.concatenate((self.link_states, new_routing_table))

        return new_state

    def reset(self):
        # Reset the network environment
        self.net.stop()
        self.net = self.setup_network()
        return self.get_state()

    def close(self):
        # Close the network environment
        self.net.stop()

    def get_routing_table(self):
        """
        获取当前网络环境的路由表信息。

        返回:
            list[dict]: 一个包含路由表条目的列表,每个条目都是一个字典,包含以下键值对:
                - 'in_port': 输入端口
                - 'eth_dst': 目的MAC地址
                - 'out_port': 输出端口
        """
        routing_table = []

        # 遍历网络中的所有交换机
        for switch in self.net.switches:
            # 获取交换机的 OpenFlow 协议版本和解析器
            ofproto = switch.ofproto
            parser = switch.ofproto_parser

            # 发送 OpenFlow 流表查询消息
            msg = parser.OFPFlowStatsRequest(switch)
            flow_stats = switch.send_msg(msg)

            # 解析流表项,并添加到路由表中
            for stat in flow_stats.body:
                if stat.priority == 1:  # 只处理优先级为 1 的流表项
                    routing_table.append({
                        'in_port': stat.match['in_port'],
                        'eth_dst': stat.match['eth_dst'],
                        'out_port': stat.instructions[0].actions[0].port
                    })

        return routing_table

    def step(self, action):
        """
        Execute the given action and update the environment.

        Args:
            action (int): The index of the routing table entry to update.

        Returns:
            next_state (numpy.ndarray): The new state of the environment.
            reward (float): The reward for the taken action.
            done (bool): Whether the episode has ended.
        """
        # Update the routing table based on the action
        new_state = self.update_state(action)

        # Update the network based on the new state
        self.update_network(new_state)

        # Calculate the reward for the new state
        reward = self.get_reward(new_state)

        # Check if the episode has ended
        done = self.check_episode_end()

        return new_state, reward, done

    def get_current_time(self):
        """
        Get the current time since the environment was created.

        Returns:
            current_time (float): The time in seconds since the environment was created.
        """
        return time.time() - self.start_time

    def update_network(self, new_state):
        """
        Update the Mininet network based on the new state.

        Args:
            new_state (numpy.ndarray): The new state of the environment.
        """
        # Implement the logic to update the Mininet network based on the new state
        pass

    def check_episode_end(self):
        """
        Check if the current episode has ended.

        Returns:
            done (bool): Whether the episode has ended.
        """
        # Implement the logic to determine if the episode has ended
        return False