"""Experiments on the effect of the learning rate on the stochastic method, for the basic cases."""
import json
from multiprocessing import Pool

from tqdm import tqdm

from clustering_techniques import compute_matching_norm
from experiment_utils import run_parametrised_stochastic_iteration
from toy_datasets import generate_G3, generate_G6, generate_C2, generate_bp


def pooled_iteration_function(iteration):
    """Used for the multiprocessing Pool of run_iterations."""
    return run_parametrised_stochastic_iteration(graph_function, (c,), random_state=iteration, learning_rates=learning_rates)

if __name__ == '__main__':
    # Experiment parameters
    graphs = list(zip(["C2", "G3", "G6"], [generate_C2, generate_G3, generate_G6]))
    cluster_sizes = [5, 10, 20, 40]
    learning_rates = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
    range_of_iterations = range(100)

    # Experiment results dict
    results = {graph_triplet[0]: {} for graph_triplet in graphs}

    for graph_name, graph_function in graphs:
        for c in cluster_sizes:
            print("Experiment on {} with c={}".format(graph_name, c))

            experiment_measures = {"template_sto_{}".format(lr): {"ari": [], "projector_distance": [], "time": []}
                                   for lr in learning_rates}

            with Pool(20) as pool:
                iteration_measures_list = list(tqdm(pool.imap(pooled_iteration_function, range_of_iterations), total=len(range_of_iterations)))
                #for iteration in tqdm(range_of_iterations):
                #    iteration_measures = run_iteration(graph_function, (c,), random_state=iteration)

                for iteration_measures in iteration_measures_list:
                    for method in ["template_sto_{}".format(lr) for lr in learning_rates]:
                        for measure in ["ari", "projector_distance", "time"]:
                            experiment_measures[method][measure].append(iteration_measures[method][measure])

            results[graph_name][c] = experiment_measures

            with open("alpha_results.json", "w") as fp:
                json.dump(results, fp, indent=2)
