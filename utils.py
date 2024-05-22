import networkx as nx
import numpy as np


def parse_network_topology(topo_file):
    """
    Parse network topology from a file and return a NetworkX graph.

    Args:
        topo_file (str): Path to the topology file.

    Returns:
        nx.Graph: A NetworkX graph representing the network topology.
    """
    G = nx.Graph()
    with open(topo_file, 'r') as f:
        for line in f:
            node1, node2 = line.strip().split()
            G.add_edge(node1, node2)
    return G

def get_network_topology(topo_file):
    """
    获取网络拓扑作为一个NetworkX图。
    参数:
        topo_file (str): 拓扑文件的路径。
    返回:
        nx.Graph: 代表网络拓扑的NetworkX图。
    """
    return parse_network_topology(topo_file)


def calculate_link_metrics(G, link_params):
    """
    Calculate link metrics for a given NetworkX graph.

    Args:
        G (nx.Graph): A NetworkX graph representing the network topology.
        link_params (dict): A dictionary of link parameters, e.g., {'bandwidth': (10, 100), 'delay': (10, 50), 'loss': (0, 0.1)}.

    Returns:
        dict: A dictionary of link metric values, e.g., {'bandwidth': np.ndarray, 'delay': np.ndarray, 'loss': np.ndarray}.
    """
    link_metrics = {}
    for metric, param_range in link_params.items():
        min_value, max_value = param_range
        link_metric_values = []
        for u, v in G.edges():
            # Calculate the link metric based on the specified metric and parameter range
            if metric == 'bandwidth':
                metric_value = min_value + (max_value - min_value) * (G.edges[(u, v)]['bandwidth_factor'])
            elif metric == 'delay':
                metric_value = min_value + (max_value - min_value) * (G.edges[(u, v)]['delay_factor'])
            elif metric == 'loss':
                metric_value = min_value + (max_value - min_value) * (G.edges[(u, v)]['loss_factor'])
            else:
                raise ValueError(f"Invalid metric: {metric}")
            link_metric_values.append(metric_value)
        link_metrics[metric] = np.array(link_metric_values)
    return link_metrics


def create_network_graph(topo_file, link_params):
    """
    Create a NetworkX graph from a topology file and add link metric factors.

    Args:
        topo_file (str): Path to the topology file.
        link_params (dict): A dictionary of link parameters, e.g., {'bandwidth': (10, 100), 'delay': (10, 50), 'loss': (0, 0.1)}.

    Returns:
        nx.Graph: A NetworkX graph representing the network topology.
    """
    G = nx.Graph()
    with open(topo_file, 'r') as f:
        for line in f:
            node1, node2 = line.strip().split()
            G.add_edge(node1, node2, bandwidth_factor=np.random.uniform(), delay_factor=np.random.uniform(),
                       loss_factor=np.random.uniform())
    return G


def dijkstra_shortest_path(G, source, target, link_metrics):
    """
    Find the shortest path between two nodes in a weighted graph using Dijkstra's algorithm.

    Args:
        G (nx.Graph): A NetworkX graph representing the network topology.
        source (str): The source node.
        target (str): The target node.
        link_metrics (dict): A dictionary of link metric values, e.g., {'bandwidth': np.ndarray, 'delay': np.ndarray, 'loss': np.ndarray}.

    Returns:
        list: The shortest path as a list of nodes.
    """
    path = []
    for metric, values in link_metrics.items():
        path = nx.dijkstra_path(G, source, target, weight=lambda u, v, d: -d[metric])
        if path:
            break
    return path


def calculate_path_metrics(G, path, link_metrics):
    """
    Calculate the overall metrics for a given path.

    Args:
        G (nx.Graph): A NetworkX graph representing the network topology.
        path (list): A list of nodes representing the path.
        link_metrics (dict): A dictionary of link metric values, e.g., {'bandwidth': np.ndarray, 'delay': np.ndarray, 'loss': np.ndarray}.

    Returns:
        dict: A dictionary of path metrics, e.g., {'bandwidth': float, 'delay': float, 'loss': float}.
    """
    path_metrics = {}
    for metric, values in link_metrics.items():
        link_metric_values = [values[G.edges().index((path[i], path[i + 1]))] for i in range(len(path) - 1)]
        path_metrics[metric] = np.min(link_metric_values)
    return path_metrics