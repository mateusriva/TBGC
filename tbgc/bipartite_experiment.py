"""Runs the Bipartite and Hub Comparison experiment."""
import json
from multiprocessing import Pool

from tqdm import tqdm

from clustering_techniques import compute_matching_norm
from experiment_utils import run_iteration
from toy_datasets import generate_G3, generate_G6, generate_C2, generate_bp


def pooled_iteration_function(iteration):
    """Used for the multiprocessing Pool of run_iterations."""
    return run_iteration(generate_C2, (c, intra_cluster_prob, inter_cluster_prob), random_state=iteration)

if __name__ == '__main__':
    # Experiment parameters
    graphs = list(zip(["HUB", "BIPARTITE"], [0.5, 0]))
    inter_cluster_probs = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
    range_of_iterations = range(150)
    cs = [10, 20, 40]

    results = {graph_tuple[0]: {c: {} for c in cs} for graph_tuple in graphs}

    for graph_name, intra_cluster_prob in graphs:
        for inter_cluster_prob in inter_cluster_probs:
            for c in cs:
                print("Experiment on {}, with c={}, at inter_cluster_prob={}".format(graph_name, c, inter_cluster_prob))

                experiment_measures = {"template_adj": {"ari": [], "projector_distance": [], "time": []},
                                       "template_lap": {"ari": [], "projector_distance": [], "time": []},
                                       "spectral": {"ari": [], "projector_distance": [], "time": []},
                                       "modularity": {"ari": [], "projector_distance": [], "time": []},
                                       "modularity_louv": {"ari": [], "projector_distance": [], "time": []}}

                with Pool(8) as pool:
                    iteration_measures_list = list(tqdm(pool.imap(pooled_iteration_function, range_of_iterations), total=len(range_of_iterations)))

                    for iteration_measures in iteration_measures_list:
                        #for method in ["template_adj", "template_lap", "spectral", "modularity"]:
                        for method in ["modularity_louv"]:
                            for measure in ["ari", "projector_distance", "time"]:
                                experiment_measures[method][measure].append(iteration_measures[method][measure])

                results[graph_name][c][inter_cluster_prob*100] = experiment_measures

    with open("bp_results.json", "w") as fp:
        json.dump(results, fp, indent=2)