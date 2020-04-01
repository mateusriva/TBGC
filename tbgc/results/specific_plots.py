from toy_datasets import generate_bp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from real_datasets import load_email_dataset

# # plt.figure(figsize=(16,10), frameon=False)
# #for subplot, cluster_probabilities in enumerate([(0.0, 0.3), (0.0, 0.6), (0.0, 0.9)]):
#     # intra_cluster_prob, inter_cluster_prob = cluster_probabilities
#     #
#     # b, c = 2, 8
#     #
#     # G_M, M, L_M, G_O, O, L_O, vertex_labels = generate_bp(c, intra_cluster_prob, inter_cluster_prob)
#     #
#     # # np.set_printoptions(threshold=np.inf,linewidth=np.inf,suppress=True)
#     # print(intra_cluster_prob)
#     # # print(O)
#     # # print("")
#     # n_intra = np.count_nonzero(O[c:2 * c, c:2 * c]) + np.count_nonzero(O[2 * c:3 * c, 2 * c:3 * c])
#     # n_inter = np.count_nonzero(O[c:2 * c, 2 * c:3 * c]) + np.count_nonzero(O[2 * c:3 * c, c:2 * c])
#     # #print("ratio: {:.2f} ({}/{}) > true probs {:.2f},{:.2f}".format(n_intra / n_inter, n_intra, n_inter,
#     # #                                                                n_intra / (n_intra + n_inter + 20),
#     # #                                                                n_inter / (n_intra + n_inter + 20)))
#     # # print("")
#     # # np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)
#     #
#     # # Dictionary of node positions - side by side
#     # pos_of_one = np.array(
#     #     [[0, 0.125],[0, 0.25],[0, 0.375],[0, 0.5],[0, 0.625],[0, 0.75],[0, 0.875],[0, 1]])
#     # pos = np.concatenate([pos_of_one, pos_of_one + [1,0]])
#     #
#     # # plt.subplot(1,5,subplot+1)
#     # plt.figure(figsize=(4, 5))
#     # # Plotting intra edges
#     # edgelist = []
#     # for edge in G_O.edges():
#     #     x, y = edge
#     #     if vertex_labels[x] == vertex_labels[y]:
#     #         edgelist.append(edge)
#     # nx.draw_networkx(G_O, pos=dict(enumerate(pos)), node_color=vertex_labels, cmap="gist_earth", with_labels=False,
#     #                  node_size=100, edgelist=edgelist, edge_color=(0.8, 0.8, 0.8))
#     # # Plotting inter edges
#     # edgelist = []
#     # for edge in G_O.edges():
#     #     x, y = edge
#     #     if vertex_labels[x] != vertex_labels[y]:
#     #         edgelist.append(edge)
#     # nx.draw_networkx(G_O, pos=dict(enumerate(pos)), node_color=vertex_labels, vmax=1.5, cmap="gist_earth",
#     #                  with_labels=False, node_size=100, edgelist=edgelist,edge_color=(0.5,0.5,1), style="dashed")
#     #
#     # # plt.title("Probs: {}-{}%".format(intra_cluster_prob*100,inter_cluster_prob*100))
#     # plt.axis("off")
#     #plt.show()
#     #plt.savefig("{:0.0f}-{:0.0f}.eps".format(intra_cluster_prob*100,inter_cluster_prob*100))
#     #plt.clf()
# # plt.figure(figsize=(16,10), frameon=False)
# for subplot, cluster_probabilities in enumerate([(0.5, 0.3), (0.5, 0.6), (0.5, 0.9)]):
#     intra_cluster_prob, inter_cluster_prob = cluster_probabilities
#
#     b, c = 2, 8
#
#     G_M, M, L_M, G_O, O, L_O, vertex_labels = generate_bp(c, intra_cluster_prob, inter_cluster_prob)
#
#     # np.set_printoptions(threshold=np.inf,linewidth=np.inf,suppress=True)
#     print(intra_cluster_prob)
#     # print(O)
#     # print("")
#     n_intra = np.count_nonzero(O[c:2 * c, c:2 * c]) + np.count_nonzero(O[2 * c:3 * c, 2 * c:3 * c])
#     n_inter = np.count_nonzero(O[c:2 * c, 2 * c:3 * c]) + np.count_nonzero(O[2 * c:3 * c, c:2 * c])
#     #print("ratio: {:.2f} ({}/{}) > true probs {:.2f},{:.2f}".format(n_intra / n_inter, n_intra, n_inter,
#     #                                                                n_intra / (n_intra + n_inter + 20),
#     #                                                                n_inter / (n_intra + n_inter + 20)))
#     # print("")
#     # np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)
#
#     # Dictionary of node positions - concentric 8-stars
#     pos_of_one = np.array(
#         [[-0.33,-1],[-1,-0.33],[-1,0.33],[-0.33,1],[0.33,-1],[1,-0.33],[1,0.33],[0.33,1]])
#     pos = np.concatenate([pos_of_one + np.random.rand(*pos_of_one.shape)*0.3, pos_of_one*2])
#
#     # plt.subplot(1,5,subplot+1)
#     plt.figure(figsize=(4, 4))
#     # Plotting intra edges
#     edgelist = []
#     for edge in G_O.edges():
#         x, y = edge
#         if vertex_labels[x] == vertex_labels[y]:
#             edgelist.append(edge)
#     nx.draw_networkx(G_O, pos=dict(enumerate(pos)), node_color=vertex_labels, cmap="gist_earth", with_labels=False,
#                      node_size=100, edgelist=edgelist, edge_color=(1, 0.5, 0.5))
#     # Plotting inter edges
#     edgelist = []
#     for edge in G_O.edges():
#         x, y = edge
#         if vertex_labels[x] != vertex_labels[y]:
#             edgelist.append(edge)
#     nx.draw_networkx(G_O, pos=dict(enumerate(pos)), node_color=vertex_labels, vmax=1.5, cmap="gist_earth",
#                      with_labels=False, node_size=100, edgelist=edgelist,edge_color=(0.5,0.5,1), style="dashed")
#
#     # plt.title("Probs: {}-{}%".format(intra_cluster_prob*100,inter_cluster_prob*100))
#     plt.axis("off")
#     #plt.show()
#     plt.savefig("{:0.0f}-{:0.0f}.eps".format(intra_cluster_prob*100,inter_cluster_prob*100))
#     plt.clf()
# # from google.colab import files
# # plt.savefig("probs.png")
# # files.download("probs.png")
import json

from tqdm import tqdm

from clustering_techniques import compute_matching_norm
from experiment_utils import run_iteration
from toy_datasets import generate_G3, generate_G6, generate_C2, generate_bp
from sklearn.metrics import adjusted_rand_score

from clustering_techniques import cluster_template, cluster_spectral, cluster_modularity
from clustering_techniques import compute_matching_norm

np.random.seed(0)

# Generate graph
G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels = generate_G6(10)
# Computing ground-truth P from vertex labels
k, n = A_M.shape[0], A_O.shape[0]  # Getting cluster size (amount of vertices in model) and observation size
P_gt = np.zeros((n, k))
P_gt[np.arange(n), vertex_labels] = 1
P_gt = P_gt @ np.diag(1 / np.sqrt(np.diag(P_gt.T @ P_gt)))  # TODO: is this still correct on adjacency?
# w/ division ^
# template_adj 0.8404879031765299
# template_lap 1.1156126900660381
# spectral 1.4298900861891668
# wo/ division
# template_adj 18.157258517397675
# template_lap 18.576717569039133
# spectral 18.576468690032076

# Run techniques
template_adj_prediction, P_opt_template_adj, template_adj_time = \
    cluster_template(A_M, A_O, mode="adjacency")
spectral_prediction, P_opt_spectral, spectral_time = \
    cluster_spectral(A_M, A_O)
modularity_prediction, _, modularity_time = \
    cluster_modularity(A_M, A_O)

# Compute measures
measures = {"template_adj": {}, "spectral": {}, "modularity": {}}
for method, prediction, features, times in zip(["template_adj", "spectral", "modularity"],
                                               [template_adj_prediction, spectral_prediction,
                                                modularity_prediction],
                                               [P_opt_template_adj, P_opt_spectral, None],
                                               [template_adj_time, spectral_time, modularity_time]):
    # Adjusted Rand Index
    measures[method]["ari"] = adjusted_rand_score(vertex_labels, prediction)
    # Distance to gt projector
    if method != "modularity":
        measures[method]["projector_distance"] = np.linalg.norm(
            np.matmul(P_gt, P_gt.T) - np.matmul(features, features.T))
    else:
        measures[method]["projector_distance"] = "N/A"
    # Computational time
    measures[method]["time"] = times

    plt.figure(figsize=(8,3))
    plt.axis("off")
    # A ten-star positionining
    from math import sqrt
    ten_star_positions = np.array([(0,1), (1/2,sqrt(3)/2), (sqrt(3)/2,1/2), (sqrt(3)/2,-1/2),(1/2,-sqrt(3)/2),
                          (0,-1), (-1/2,sqrt(3)/2), (-sqrt(3)/2,1/2), (-sqrt(3)/2,-1/2),(-1/2,-sqrt(3)/2)])
    all_positions = dict(enumerate(np.concatenate([ten_star_positions,
                                    ten_star_positions + [2.5,0],
                                    ten_star_positions + [5,0],
                                    ten_star_positions + [3.75,3],
                                    ten_star_positions + [7.5,0],
                                    ten_star_positions + [10,0]])))
    nx.draw_networkx(G_O, with_labels=False, node_color=prediction, cmap="Set2", vmax=7, pos=all_positions)
    print("Measures of", method, ":", measures[method])
    plt.tight_layout()
    plt.savefig("qualitative_G6_{}.eps".format(method))
    plt.clf()
    #plt.show()
    