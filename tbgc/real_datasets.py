"""Downloads and load real graphs."""
import os
import gzip

import numpy as np
import networkx as nx
import wget


def load_email_dataset(cut=None, noise=0):
    """Loads the email-Eu-core dataset, along with its base communities.

    Parameters
    ----------
    cut : None, int
        Whether to cut the graph. If int, cuts neighbors above `cut`
    noise : int
        Std of mean-zero gaussian noise to add to model. Default: zero"""
    # Checking if present, downloading if not
    if not os.path.exists("data/email-Eu-core.txt.gz"):
        wget.download("https://snap.stanford.edu/data/email-Eu-core.txt.gz", out="data/email-Eu-core.txt.gz")
    if not os.path.exists("data/email-Eu-core-department-labels.txt.gz"):
        wget.download("https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz", out="data/email-Eu-core-department-labels.txt.gz")

    # Reading graph
    # email-Eu-core is stored as a set of edges, with no header, space-delimited and a trailing newline
    with gzip.open("data/email-Eu-core.txt.gz") as fp:
        loaded_lines = fp.readlines()
    list_of_tuples = [tuple(map(int, line.decode("ascii").strip().split(" "))) for line in loaded_lines]
    graph_email = nx.Graph(list_of_tuples)

    # Reading labels
    # email-Eu-core-department-labels is stored as a set of labels, each line is "<node> <label>\n"
    with gzip.open("data/email-Eu-core-department-labels.txt.gz") as fp:
        loaded_lines = fp.readlines()
    list_of_tuples = [tuple(map(int, line.decode("ascii").strip().split(" "))) for line in loaded_lines]
    dict_of_labels = dict(list_of_tuples)

    A_M, A_O, vertex_labels = process_loaded_graph(graph_email, dict_of_labels, cut)

    # Adding noise
    if noise > 0:
        A_M = (A_M + np.random.normal(0, noise, A_M.shape)).clip(min=0.0)

    return None, A_M, None, None, A_O, None, vertex_labels


def load_dbpl_dataset(cut=None):
    """Loads the dbpl dataset and its base communities"""
    # Checking if present, downloading if not
    if not os.path.exists("data/com-dblp.ungraph.txt.gz"):
        wget.download("https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz", out="data/com-dblp.ungraph.txt.gz")
    if not os.path.exists("data/com-dblp.all.cmty.txt.gz"):
        wget.download("https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz", out="data/com-dblp.all.cmty.txt.gz")

    # Reading graph
    # dbpl is stored as a set of edges, with 4-lines header, tab-delimited and a trailing newline
    with gzip.open("data/com-dblp.ungraph.txt.gz") as fp:
        loaded_lines = fp.readlines()
    list_of_tuples = [tuple(map(int, line.decode("ascii").strip().split("\t"))) for line in loaded_lines[4:]]
    loaded_graph = nx.Graph(list_of_tuples)

    # Reading labels
    # dbpl cmty is stored as a set of communities, each line is "<node1>\t<node2>\t<node3>...\n"
    with gzip.open("data/com-dblp.all.cmty.txt.gz") as fp:
        loaded_lines = fp.readlines()
    list_of_communities = [(map(int, line.decode("ascii").strip().split("\t"))) for line in loaded_lines]
    dict_of_labels = {}
    for i, community in enumerate(list_of_communities):
        for node in community:
            dict_of_labels[node] = i

    # Not *all* nodes have a community. To fix that, we're adding them to the "last" community
    last_community = len(list_of_communities)
    for node in loaded_graph:
        try:
            dict_of_labels[node]
        except KeyError:
            dict_of_labels[node] = last_community

    A_M, A_O, vertex_labels = process_loaded_graph(loaded_graph, dict_of_labels, cut)

    return None, A_M, None, None, A_O, None, vertex_labels


def load_custom_dbpl_dataset(cut=None):
    """Loads the dbpl dataset and its base communities"""
    # Reading graph
    # dbpl cutom is stored as a set of edges, with no lines header, space-delimited and a trailing newline
    with open("data_custom/com-dblp.8cut.ungraph.txt") as fp:
        loaded_lines = fp.readlines()
    list_of_tuples = [tuple(map(int, line.strip().split(" "))) for line in loaded_lines[4:]]
    loaded_graph = nx.Graph(list_of_tuples)

    # Reading labels
    # dbpl cmty is stored as a set of communities, each line is "<node1>\t<node2>\t<node3>...\n"
    with open("data_custom/com-dblp.8cut.cmty.txt") as fp:
        loaded_lines = fp.readlines()
    list_of_tuples = [tuple(map(int, line.strip().split(" "))) for line in loaded_lines]
    dict_of_labels = dict(list_of_tuples)

    A_M, A_O, vertex_labels = process_loaded_graph(loaded_graph, dict_of_labels, cut)

    return None, A_M, None, None, A_O, None, vertex_labels


def process_loaded_graph(loaded_graph, dict_of_labels, cut=None, return_connected=True):
    # Cutting nodes
    allowed_nodes, cut_nodes = [], []
    if cut is not None:
        for i, node in enumerate(loaded_graph):
            if i < cut: allowed_nodes.append(node)
            else: cut_nodes.append(node)
        for cut_node in cut_nodes:
            loaded_graph.remove_node(cut_node)

    # Cutting labels
    if cut is not None:
        for cut_node in cut_nodes:
            del dict_of_labels[cut_node]

    # Cutting to connected component
    if return_connected:
        ccs = list(nx.connected_components(loaded_graph))
        if len(ccs) > 1:
          out_of_cc = list(set().union(*(ccs[1:])))
          loaded_graph.remove_nodes_from(out_of_cc)
          for i in out_of_cc:
              del dict_of_labels[int(i)]

    # 'Compressing' the graph nodes
    node_conversion_dict = { node : i for i, node in enumerate(loaded_graph)}

    compressed_graph = nx.relabel_nodes(loaded_graph, node_conversion_dict, copy=True)

    label_compression_dict = {v:k for k,v in enumerate(np.unique(list(dict_of_labels.values())))}
    compressed_labels = [label_compression_dict[dict_of_labels[node]] for node in loaded_graph]

    # Building observation graph matrix
    A_O = nx.to_numpy_matrix(compressed_graph)

    # Building model graph matrix
    A_M = np.zeros([len(np.unique(compressed_labels))]*2)
    for node in compressed_graph:
        for neighbor in nx.neighbors(compressed_graph, node):
            community_from, community_to = compressed_labels[node], compressed_labels[neighbor]
            A_M[community_from, community_to]  = A_M[community_from, community_to] + 1

    return A_M, A_O, compressed_labels


def school_label_process(item):
    if item == "cm1a": return 0
    if item == "ce1a": return 1
    if item == "cm1b": return 2
    if item == "ce1b": return 3
    if item == "ce2a": return 4
    if item == "cm2a": return 5
    if item == "ce2b": return 6
    if item == "cm2b": return 7
    if item == "teachers": return 8
    if item == "cpa": return 9
    if item == "cpb": return 10
    else: return int(item)


def load_school_dataset(which=1, cut=None, noise=0):
    """Loads the school face to face interaction dataset."""
    assert which in [1,2], "dataset must be 1 or 2"
    # school dataset is stored as gml data
    loaded_graph = nx.read_gml("data/school{}.gml".format(which), label=None).to_undirected()

    # Reading labels
    # school communities are stored as pairs of node_id and labels
    with open("data/school_labels.txt") as fp:
        loaded_lines = fp.readlines()
    list_of_tuples = [tuple(map(school_label_process, line.strip().split("\t"))) for line in loaded_lines]
    dict_of_labels = dict(list_of_tuples)

    A_M, A_O, vertex_labels = process_loaded_graph(loaded_graph, dict_of_labels, cut)

    # Adding noise
    if noise > 0:
        A_M = (A_M + np.random.normal(0, noise, A_M.shape)).clip(min=0.0)

    return None, A_M, None, None, A_O, None, vertex_labels


if __name__ == '__main__':
    _, m, _, _, o, _, l = load_school_dataset(cut=None)