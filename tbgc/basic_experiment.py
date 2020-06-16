"""Runs the Basic Comparison Experiment."""
import json
from multiprocessing import Pool

from tqdm import tqdm

from clustering_techniques import compute_matching_norm
from experiment_utils import run_iteration
from toy_datasets import generate_G3, generate_G6, generate_C2, generate_bp


def pooled_iteration_function(iteration):
    """Used for the multiprocessing Pool of run_iterations."""
    return run_iteration(graph_function, (c,), random_state=iteration)

if __name__ == '__main__':
    # Experiment parameters
    graphs = list(zip(["C2", "G3", "G6"], [generate_C2, generate_G3, generate_G6]))
    cluster_sizes = [5, 10, 20, 40, 80]
    range_of_iterations = range(100)

    # Experiment results dict
    results = {graph_triplet[0]: {} for graph_triplet in graphs}

    for graph_name, graph_function in graphs:
        for c in cluster_sizes:
            print("Experiment on {} with c={}".format(graph_name, c))

            experiment_measures = {"template_adj": {"ari": [], "projector_distance": [], "time": []},
                                   "template_lap": {"ari": [], "projector_distance": [], "time": []},
                                   "template_sto": {"ari": [], "projector_distance": [], "time": []},
                                   "template_stok": {"ari": [], "projector_distance": [], "time": []},
                                   "spectral": {"ari": [], "projector_distance": [], "time": []},
                                   "modularity": {"ari": [], "projector_distance": [], "time": []},
                                   "modularity_louv": {"ari": [], "projector_distance": [], "time": []}}

            with Pool(8) as pool:
                iteration_measures_list = list(tqdm(pool.imap(pooled_iteration_function, range_of_iterations), total=len(range_of_iterations)))
                #for iteration in tqdm(range_of_iterations):
                #    iteration_measures = run_iteration(graph_function, (c,), random_state=iteration)

                for iteration_measures in iteration_measures_list:
                    for method in ["template_adj", "template_sto", "template_stok", "spectral", "modularity", "modularity_louv"]:
                        for measure in ["ari", "projector_distance", "time"]:
                            experiment_measures[method][measure].append(iteration_measures[method][measure])

            results[graph_name][c] = experiment_measures

    with open("basic_results.json", "w") as fp:
        json.dump(results, fp, indent=2)