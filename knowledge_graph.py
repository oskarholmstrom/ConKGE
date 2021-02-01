
import networkx as nx



class KnowledgeGraph(object):
    """ 
    A knowledge graph class that contains a multi-directed graph object,
    and exists as an intermediary interface to that object.
    """


    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.num_triples = 0

    def add_triple(self, triple):
        # Adds an edge with the end-nodes triple[0] and triple[2], with relation as edge data
        self.graph.add_edge(triple[0], triple[2], relation=[triple[1]])
    
    def get_nodes(self):
        return self.graph.nodes()

    def get_triples(self):
        return self.graph.edges(data='relation')

    def get_neighbors(self, node):
        neighbors = self.graph.out_edges(node, data=True)
        neighbors_list = []
        for n in neighbors:
                neighbors_list.append((n[1], n[2]['relation'][0]))
        return neighbors_list


    def get_adj_nodes(self, node):
        adj_nodes = {}
        for node, relations in self.graph.adj[node].items():
            adj_nodes[node] = []
            for key in relations:
                adj_nodes[node].append(relations[key]['relation'][0])

        return adj_nodes