from utils import visualize_graph, visualize_embedding
from dataset import HW3Dataset
from torch_geometric.utils import to_networkx

if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    print(data)
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.shape[0]}')
    print(f'Training node label rate: {int(data.train_mask.shape[0]) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)