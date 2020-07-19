"""Plots JSON results file from the TBGC experiments."""
import json

from math import log10
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_basic_results():# Plotting basic results
    graph_names = ["C2","G3","G6"]
    measures = ["ari", "projector_distance", "time"]
    #techniques = ["template_adj", "template_lap", "spectral", "modularity"]
    techniques = ["template_adj", "template_sto", "spectral", "modularity", "modularity_louv"]
    cluster_sizes = ["5","10","20","40","80"]

    with open("basic_results.json") as fp:
        results_dict = json.load(fp)

    # BASIC PLOT SCHEMA: one plot per measure and graph type, x-axis is cluster size, lines are techniques
    for graph_name in graph_names:
        # Setting up printables name
        if graph_name == "G3": graph_print_name = "$G_3$"
        if graph_name == "G6": graph_print_name = "$G_6$"
        if graph_name == "C2": graph_print_name = "$C2$"

        for measure in measures:
            # Setting up printables name
            if measure == "ari": measure_print = "ARI"
            if measure == "projector_distance": measure_print = "Projector Distance"
            if measure == "time": measure_print = "Time"

            # Starting plot
            plt.figure(figsize=(10,4))
            plt.rcParams.update({'font.size': 18})
            plt.title(graph_print_name)
            for technique in techniques:
                # Skipping unwelcome techniques
                if technique in ["template_sto", "template_stok"]:
                    continue
                # Skipping modularity and PD
                if (technique == "modularity" or technique == "modularity_louv") and measure == "projector_distance":
                    continue
                # Setting up printables name
                if technique == "template_adj": technique_printable = "Template-based" ; color = (0.6,0,0) ; marker = "^" ; variation = -0.15
                #if technique == "template_sto": technique_printable = "Stochastic TB" ; color = (1.0,0.5,0.1) ; marker = "v" ; variation = -0.075
                #if technique == "template_stok": technique_printable = "STB + $k$-means" ; color = (0.8,0.3,0.1) ; marker = ">" ; variation = 0.00
                if technique == "spectral": technique_printable = "Spectral" ; color = "g" ; marker = "o" ; variation = -0.075
                if technique == "modularity": technique_printable = "CNM Modularity" ; color = "b" ; marker = "D" ; variation = 0.075
                if technique == "modularity_louv": technique_printable = "Louvain" ; color = (0.6,0.0,0.6) ; marker = "*" ; variation = 0.15

                measure_list = [results_dict[graph_name][cluster_size][technique][measure] for cluster_size in cluster_sizes]

                plt.errorbar(x=np.array(range(len(cluster_sizes))) + variation,
                             y=np.mean(measure_list, axis=1),
                             yerr=np.std(measure_list, axis=1),
                             markersize=6, marker=marker, color=color, capsize=2,
                             label=technique_printable)

            plt.xticks(range(len(cluster_sizes)), labels=cluster_sizes)
            plt.ylabel(measure_print)
            plt.xlabel("Cluster size")
            plt.legend()
            plt.grid(b=True, axis="both", which="major")

            plt.tight_layout()
            plt.savefig("basic_{}_{}.eps".format(graph_name, measure))


def plot_alpha_results():# Plotting alpha results
    graph_names = ["C2","G3","G6"]
    measures = ["ari", "projector_distance", "time"]
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    cluster_sizes = ["5","10","20","40"]

    with open("alpha_results.json") as fp:
        results_dict = json.load(fp)

    # BASIC PLOT SCHEMA: one plot per measure and graph type, x-axis is cluster size, lines are techniques
    for graph_name in graph_names:
        # Setting up printables name
        if graph_name == "G3": graph_print_name = "$G_3$"
        if graph_name == "G6": graph_print_name = "$G_6$"
        if graph_name == "C2": graph_print_name = "$C2$"

        for measure in measures:
            # Setting up printables name
            if measure == "ari": measure_print = "ARI"
            if measure == "projector_distance": measure_print = "Projector Distance"
            if measure == "time": measure_print = "Time"

            # Starting plot
            plt.figure(figsize=(10,8))
            plt.rcParams.update({'font.size': 18})
            plt.title("Stochastic TBGC on {}".format(graph_print_name))
            cm = plt.get_cmap("jet")
            for idx, lr in enumerate(learning_rates):
                technique = "template_sto_{}".format(lr)
                # Setting up printables name
                technique_printable = "$\\alpha = {}$".format(lr)
                color = cm(idx/(len(learning_rates)))
                variation = (idx - ((len(learning_rates)+1)/2))/(len(learning_rates)*1.6)
                #if technique == "template_adj": technique_printable = "Template-based" ; color = (0.6,0,0) ; marker = "^" ; variation = -0.15
                #if technique == "template_sto": technique_printable = "Stochastic TB" ; color = (1.0,0.5,0.1) ; marker = "v" ; variation = -0.05
                #if technique == "spectral": technique_printable = "Spectral" ; color = "g" ; marker = "o" ; variation = 0.05
                #if technique == "modularity": technique_printable = "Modularity" ; color = "b" ; marker = "D" ; variation = 0.15

                measure_list = [results_dict[graph_name][cluster_size][technique][measure] for cluster_size in cluster_sizes]

                plt.errorbar(x=np.array(range(len(cluster_sizes))) + variation,
                             y=np.mean(measure_list, axis=1),
                             yerr=np.std(measure_list, axis=1),
                             markersize=6, marker=".", color=color, capsize=2,
                             label=technique_printable)

            plt.xticks(range(len(cluster_sizes)), labels=cluster_sizes)
            plt.ylabel(measure_print)
            plt.xlabel("Cluster size")
            plt.legend()
            plt.grid(b=True, axis="both", which="major")

            plt.tight_layout()
            plt.savefig("alpha_{}_{}.eps".format(graph_name, measure))
            #plt.show()


def plot_progression_results():
    intra_cluster_probabilities = ["80.0", "75.0", "70.0", "65.0", "60.0", "55.0", "50.0", "45.0", "40.0"][::-1]
    inter_cluster_probabilities_print = ["0.20", "0.25", "0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60"][::-1]
    measures = ["ari", "projector_distance", "time"]
    #techniques = ["template_adj", "template_lap", "spectral", "modularity"]
    techniques = ["template_adj", "spectral", "modularity", "modularity_louv"]
    cluster_sizes = ["10","20","40"]

    with open("progression_results.json") as fp:
        results_dict = json.load(fp)

    # PROGRESSION PLOT SCHEMA: one plot per cluster size/measure, x-axis is intraprob, lines are techniques
    for cluster_size in cluster_sizes:
        for measure in measures:
            # Setting up printables name
            if measure == "ari": measure_print = "ARI"
            if measure == "projector_distance": measure_print = "Projector Distance"
            if measure == "time": measure_print = "Time"

            # Starting plot
            plt.figure(figsize=(10,4))
            plt.rcParams.update({'font.size': 18})
            plt.title("Cluster size {}".format(cluster_size))
            for technique in techniques:
                # Skipping modularity and PD
                if (technique == "modularity" or technique == "modularity_louv") and measure == "projector_distance":
                    continue
                # Setting up printables name
                if technique == "template_adj": technique_printable = "Template-based" ; color = (0.6,0,0) ; marker = "^" ; variation = -0.15
                #if technique == "template_sto": technique_printable = "Stochastic TB" ; color = (1.0,0.5,0.1) ; marker = "v" ; variation = -0.075
                #if technique == "template_stok": technique_printable = "STB + $k$-means" ; color = (0.8,0.3,0.1) ; marker = ">" ; variation = 0.00
                if technique == "spectral": technique_printable = "Spectral" ; color = "g" ; marker = "o" ; variation = -0.075
                if technique == "modularity": technique_printable = "CNM Modularity" ; color = "b" ; marker = "D" ; variation = 0.075
                if technique == "modularity_louv": technique_printable = "Louvain" ; color = (0.6,0.0,0.6) ; marker = "*" ; variation = 0.15

                measure_list = [results_dict[cluster_size][icp][technique][measure] for icp in
                                intra_cluster_probabilities]

                plt.errorbar(x=np.array(range(len(intra_cluster_probabilities))) + variation,
                             y=np.mean(measure_list, axis=1),
                             yerr=np.std(measure_list, axis=1),
                             markersize=6, marker=marker, color=color, capsize=2,
                             label=technique_printable)

            plt.xticks(range(len(intra_cluster_probabilities)), labels=inter_cluster_probabilities_print)
            plt.ylabel(measure_print)
            plt.xlabel("Inter-cluster Probability")
            plt.legend()
            plt.grid(b=True, axis="both", which="major")

            plt.tight_layout()
            plt.savefig("progression_{}_{}.eps".format(cluster_size, measure))


def plot_bp_results():
    inter_cluster_probabilities = ["80", "75", "70", "65", "60", "55", "50", "45", "40"]
    inter_cluster_probabilities_print = ["0.80", "0.75", "0.70", "0.65", "0.60", "0.55", "0.50", "0.45", "0.40"]
    measures = ["ari", "projector_distance", "time"]
    #techniques = ["template_adj", "template_lap", "spectral", "modularity"]
    techniques = ["template_adj", "spectral", "modularity", "modularity_louv"]
    cluster_sizes = ["10","20","40"]
    graphs = ["HUB", "BIPARTITE"]

    with open("bp_results.json") as fp:
        results_dict = json.load(fp)

    # PROGRESSION PLOT SCHEMA: one plot per graph/cluster size/measure, x-axis is interprob, lines are techniques
    for graph_name in graphs:
        # printables
        if graph_name == "HUB": graph_name_print = "Hub (0.5 ICP)"
        if graph_name == "BIPARTITE": graph_name_print = "Bipartite (0.0 ICP)"
        for cluster_size in cluster_sizes:
            for measure in measures:
                # Setting up printables name
                if measure == "ari": measure_print = "ARI"
                if measure == "projector_distance": measure_print = "Projector Distance"
                if measure == "time": measure_print = "Time"

                # Starting plot
                plt.figure(figsize=(10,4))
                plt.rcParams.update({'font.size': 18})
                plt.title("{}, cluster size {}".format(graph_name_print, cluster_size))
                for technique in techniques:
                    # Skipping modularity and PD
                    if (
                            technique == "modularity" or technique == "modularity_louv") and measure == "projector_distance":
                        continue
                    # Setting up printables name
                    if technique == "template_adj": technique_printable = "Template-based"; color = (
                    0.6, 0, 0); marker = "^"; variation = -0.15
                    # if technique == "template_sto": technique_printable = "Stochastic TB" ; color = (1.0,0.5,0.1) ; marker = "v" ; variation = -0.075
                    # if technique == "template_stok": technique_printable = "STB + $k$-means" ; color = (0.8,0.3,0.1) ; marker = ">" ; variation = 0.00
                    if technique == "spectral": technique_printable = "Spectral"; color = "g"; marker = "o"; variation = -0.075
                    if technique == "modularity": technique_printable = "CNM Modularity"; color = "b"; marker = "D"; variation = 0.075
                    if technique == "modularity_louv": technique_printable = "Louvain"; color = (
                    0.6, 0.0, 0.6); marker = "*"; variation = 0.15

                    measure_list = [results_dict[graph_name][cluster_size][icp][technique][measure] for icp in
                                    inter_cluster_probabilities]

                    plt.errorbar(x=np.array(range(len(inter_cluster_probabilities))) + variation,
                                 y=np.mean(measure_list, axis=1),
                                 yerr=np.std(measure_list, axis=1),
                                 markersize=6, marker=marker, color=color, capsize=2,
                                 label=technique_printable)

                plt.xticks(range(len(inter_cluster_probabilities)), labels=inter_cluster_probabilities_print)
                plt.ylabel(measure_print)
                plt.xlabel("Inter-cluster Probability")
                plt.legend()
                plt.grid(b=True, axis="both", which="major")

                plt.tight_layout()
                #plt.show()
                plt.savefig("bipartite_{}_{}_{}.eps".format(graph_name, cluster_size, measure))


def print_real():
    measures = ["ari", "projector_distance", "time"]
    techniques = ["template_adj", "spectral", "modularity", "modularity_louv"]
    technique_printables = ["TB", "Spectral", "CNM Modularity", "Louvain"]
    graphs = ["email", "school1", "school2"]

    with open("real_results_noisy.json") as fp:
        results_dict = json.load(fp)

    # REAL TABLE SCHEMA: header is dataset, metric, each technique; multi line for datasets
    print(" & ".join(["Dataset name", "Metric"] + technique_printables) + "\\\\\n\\hline")
    for graph in graphs:
        print("\\multirow{"+str(len(measures))+"}{*}{"+graph+"} & ", end="")
        for measure in measures:
            if measure != "ari":
                print(" & ", end="")
            if measure == "ari": measure_print = "ARI"
            if measure == "projector_distance": measure_print = "Projector Distance"
            if measure == "time": measure_print = "Time"
            print(measure_print, end="")

            for technique in techniques:
                if (technique == "modularity" or technique == "modularity_louv") and measure == "projector_distance":
                    print(" & N/A", end="")
                    continue
                measure_list = results_dict[graph]["0"][technique][measure]
                print(" & ${:.2f} \\pm {:.2f}$".format(np.mean(measure_list), np.std(measure_list)), end="")
            print("\\\\")

def plot_real_noisy():
    measures = ["ari", "projector_distance"]
    techniques = ["template_adj"]
    technique_printables = ["Template-based"]
    graphs = ["email"]
    noises=["0","1","2","5","10","20","30",]

    with open("real_results_noisy.json") as fp:
        results_dict = json.load(fp)

    # NOISY PLOT SCHEMA: x-axis is noise level. need horizontal lines for baselines!
        # ARI
        plt.figure(figsize=(10, 4))
        plt.rcParams.update({'font.size': 18})
        plt.title("ARI - Noisy Model")

        # Plotting baseline spectral line
        spectral_mean = np.mean([results_dict["email"][noise]["spectral"]["ari"] for noise in noises])
        spectral_std = np.std([results_dict["email"][noise]["spectral"]["ari"] for noise in noises])
        plt.hlines(spectral_mean, xmin=0, xmax=len(noises) - 1, color="g", linestyle="--", label="Spectral")
        # plt.fill_between(range(len(noises)), [spectral_mean + spectral_std] * len(noises),
        #                 [spectral_mean - spectral_std] * len(noises), color=(0, 1, 0, 0.3))
        # Plotting baseline modularity line
        modularity_mean = np.mean([results_dict["email"][noise]["modularity"]["ari"] for noise in noises])
        modularity_std = np.std([results_dict["email"][noise]["modularity"]["ari"] for noise in noises])
        plt.hlines(modularity_mean, xmin=0, xmax=len(noises) - 1, color="b", label="CNM Modularity", linestyle="--")
        modularity_louv_mean = np.mean([results_dict["email"][noise]["modularity_louv"]["ari"] for noise in noises])
        modularity_louv_std = np.std([results_dict["email"][noise]["modularity_louv"]["ari"] for noise in noises])
        plt.hlines(modularity_louv_mean, xmin=0, xmax=len(noises) - 1, color=(0.6,0,0.6), label="Louvain", linestyle="--")
        # plt.fill_between(range(len(noises)), [modularity_mean+modularity_std]*len(noises), [modularity_mean-modularity_std]*len(noises), color=(0,1,0,0.3))

        # Plotting the ARI
        measure_list = [results_dict["email"][noise]["template_adj"]["ari"] for noise in noises]
        plt.errorbar(x=np.array(range(len(noises))),
                     y=np.mean(measure_list, axis=1),
                     yerr=np.std(measure_list, axis=1),
                     markersize=6, marker="^", color=(0.6, 0, 0), capsize=2,
                     label="Template-Based")

        plt.ylabel("ARI")
        plt.xlabel("Model Noise $\\sigma$")
        plt.legend()
        plt.xticks(range(len(noises)), labels=noises)

        plt.grid(b=True, axis="both", which="major")

        plt.tight_layout()
        plt.savefig("email_noisy_ari.eps")
        # Projector Distance
        plt.figure(figsize=(10, 4))
        plt.rcParams.update({'font.size': 18})
        plt.title("Projector Distance - Noisy Model")

        # Plotting baseline spectral line
        spectral_mean = np.mean([results_dict["email"][noise]["spectral"]["projector_distance"] for noise in noises])
        spectral_std = np.std([results_dict["email"][noise]["spectral"]["projector_distance"] for noise in noises])
        plt.hlines(spectral_mean, xmin=0, xmax=len(noises) - 1, color="g", linestyle="--", label="Spectral")
        # plt.fill_between(range(len(noises)), [spectral_mean + spectral_std] * len(noises),
        #                 [spectral_mean - spectral_std] * len(noises), color=(0, 1, 0, 0.3))
        # Plotting baseline modularity line
        #modularity_mean = np.mean([results_dict["email"][noise]["modularity"]["projector_distance"] for noise in noises])
        #modularity_std = np.std([results_dict["email"][noise]["modularity"]["projector_distance"] for noise in noises])
        #plt.hlines(modularity_mean, xmin=0, xmax=len(noises) - 1, color="b", label="Modularity")
        # plt.fill_between(range(len(noises)), [modularity_mean+modularity_std]*len(noises), [modularity_mean-modularity_std]*len(noises), color=(0,1,0,0.3))

        # Plotting the Projector Distance
        measure_list = [results_dict["email"][noise]["template_adj"]["projector_distance"] for noise in noises]
        plt.errorbar(x=np.array(range(len(noises))),
                     y=np.mean(measure_list, axis=1),
                     yerr=np.std(measure_list, axis=1),
                     markersize=6, marker="^", color=(0.6, 0, 0), capsize=2,
                     label="Template-Based")

        plt.ylabel("Projector Distance")
        plt.xlabel("Model Noise $\\sigma$")
        plt.legend()
        plt.xticks(range(len(noises)), labels=noises)

        plt.grid(b=True, axis="both", which="major")

        plt.tight_layout()
        plt.savefig("email_noisy_pd.eps")

if __name__ == '__main__':
    plot_real_noisy()
    #print_real()
    #plot_alpha_results()
    #plot_bp_results()
    #plot_basic_results()
    #plot_progression_results()
    #plot_real_noisy()
