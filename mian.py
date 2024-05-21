from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from dqn_agent import DQNAgent
from network_env import NetworkEnv
from utils import get_network_topology, calculate_link_cost

class RyuController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RyuController, self).__init__(*args, **kwargs)
        self.agent = DQNAgent(state_size=20, action_size=50)
        self.env = NetworkEnv(get_network_topology())
        self.training_interval = 60  # 每60秒训练一次

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # 初始化网络环境和控制器
        self.env.initialize(datapath)

        # 开始主循环
        self.main_loop(datapath)

    def main_loop(self, datapath):
        last_training_time = 0
        while True:
            # 获取当前状态
            state = self.env.get_state()

            # 使用DQN代理选择动作
            action = self.agent.get_action(state)

            # 执行动作并获取下一个状态、奖励和是否完成
            next_state, reward, done = self.env.step(action)

            # 存储transition到经验池
            self.agent.remember(state, action, reward, next_state, done)

            # 更新网络状态和路由表
            self.env.update_state(next_state)
            self.update_routing_table(datapath)

            # 定期训练DQN代理
            current_time = self.env.get_current_time()
            if current_time - last_training_time >= self.training_interval:
                self.agent.replay(32)
                last_training_time = current_time

    def update_routing_table(self, datapath):
        # 根据DQN代理的决策更新交换机的路由表
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        for flow_entry in self.env.get_routing_table():
            match = parser.OFPMatch(
                in_port=flow_entry['in_port'],
                eth_dst=flow_entry['eth_dst']
            )
            actions = [parser.OFPActionOutput(flow_entry['out_port'])]
            self.add_flow(datapath, 1, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)