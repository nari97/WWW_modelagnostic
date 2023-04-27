import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from Code import ParseRules
from LatexUtils import create_regular_table, create_multi_table
import matplotlib.pyplot as plt


def create_box_plots():
    datasets = ["FB15K", "FB15K-237", "WN18", "WN18RR", "YAGO3-10"]
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(16, 4), sharey=True)
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "gray", "pink", "brown", "turquoise"]
    model_names = ["BoxE", "ComplEx", "HAKE", "HolE", "QuatE", "RotatE", "Rotpro", "TorusE", "TransE", "TuckER"]
    ctr = 0

    for dataset_name in datasets:
        folder = f"D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/BestRules/{dataset_name}"
        selecs = []

        for model in models:
            selec_array = []
            print(model)
            f = open(f"{folder}/{model}/{dataset_name}_{model}_mat_0_bestH.tsv")

            for line in f:
                if line == "\n":
                    continue

                splits = line.strip().split("\t")
                selec = float(splits[-1])
                selec_array.append(selec)
            selecs.append(selec_array)
        # print(colors)
        axs[ctr].boxplot(selecs, whis=2)
        axs[ctr].set_xticklabels(model_names, rotation=90)
        axs[ctr].set_title(f"{dataset_name}")
        ctr += 1

    # legend_labels = [f"{i+1}: {model}" for i, model in enumerate(model_names)]
    # fig.legend(labels=legend_labels,loc='upper center', bbox_to_anchor=(0.95, 0.75), ncol=1, handlelength=0)
    plt.savefig(f"D:/PhD/Work/EmbeddingInterpretibility/Interpretibility/Results/BoxPlots/all_datasets_box_bestH.png",
                bbox_inches='tight')
    plt.close()


def create_size_predictions_table():
    input = open("input.txt")
    mr_data = {'boxe': {'FB15K': {'AMR': 0.993, '|M|': 6.437, 'Ties': 0.0},
                        'FB15K237': {'AMR': 0.969, '|M|': 9.146, 'Ties': 0.0},
                        'WN18': {'AMR': 0.964, '|M|': 7.329, 'Ties': 0.0},
                        'WN18RR': {'AMR': 0.626, '|M|': 47.972, 'Ties': 0.0},
                        'YAGO3-10': {'AMR': 0.961, '|M|': 23.403, 'Ties': 0.0}},
               'complex': {'FB15K237': {'AMR': 0.927, '|M|': 21.405, 'Ties': 0.0},
                           'WN18': {'AMR': 0.952, '|M|': 9.778, 'Ties': 0.0},
                           'WN18RR': {'AMR': 0.586, '|M|': 53.128, 'Ties': 0.0},
                           'YAGO3-10': {'AMR': 0.942, '|M|': 35.166, 'Ties': 0.0},
                           'FB15K': {'AMR': 0.98, '|M|': 17.337, 'Ties': 0.0}},
               'hake': {'FB15K': {'AMR': 0.985, '|M|': 12.814, 'Ties': 0.0},
                        'FB15K237': {'AMR': 0.943, '|M|': 16.661, 'Ties': 0.0},
                        'WN18': {'AMR': 0.96, '|M|': 8.253, 'Ties': 0.0},
                        'WN18RR': {'AMR': 0.519, '|M|': 61.727, 'Ties': 0.0},
                        'YAGO3-10': {'AMR': 0.942, '|M|': 35.01, 'Ties': 0.0}},
               'hole': {'FB15K': {'AMR': 0.971, '|M|': 25.192, 'Ties': 0.0},
                        'FB15K237': {'AMR': 0.91, '|M|': 26.253, 'Ties': 0.0},
                        'WN18': {'AMR': 0.964, '|M|': 7.467, 'Ties': 0.0},
                        'WN18RR': {'AMR': 0.607, '|M|': 50.439, 'Ties': 0.0},
                        'YAGO3-10': {'AMR': 0.9, '|M|': 60.545, 'Ties': 0.0}},
               'quate': {'FB15K': {'AMR': 0.978, '|M|': 19.546, 'Ties': 0.0},
                         'FB15K237': {'AMR': 0.938, '|M|': 18.054, 'Ties': 0.0},
                         'WN18': {'AMR': 0.958, '|M|': 8.533, 'Ties': 0.0},
                         'WN18RR': {'AMR': 0.583, '|M|': 53.504, 'Ties': 0.0},
                         'YAGO3-10': {'AMR': 0.924, '|M|': 46.01, 'Ties': 0.0}},
               'rotate': {'FB15K': {'AMR': 0.985, '|M|': 13.065, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.946, '|M|': 15.702, 'Ties': 0.0},
                          'WN18': {'AMR': 0.959, '|M|': 8.3, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.705, '|M|': 37.85, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.624, '|M|': 227.967, 'Ties': 0.0}},
               'rotpro': {'FB15K': {'AMR': 0.982, '|M|': 15.486, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.895, '|M|': 30.875, 'Ties': 0.0},
                          'WN18': {'AMR': 0.973, '|M|': 5.586, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.768, '|M|': 29.732, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.716, '|M|': 172.092, 'Ties': 0.0}},
               'toruse': {'FB15K': {'AMR': 0.986, '|M|': 12.31, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.951, '|M|': 14.317, 'Ties': 0.0},
                          'WN18': {'AMR': 0.981, '|M|': 3.811, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.782, '|M|': 27.994, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.928, '|M|': 43.527, 'Ties': 0.0}},
               'transe': {'FB15K': {'AMR': 0.993, '|M|': 6.407, 'Ties': 0.0},
                          'FB15K237': {'AMR': 0.972, '|M|': 8.203, 'Ties': 0.0},
                          'WN18': {'AMR': 0.984, '|M|': 3.243, 'Ties': 0.0},
                          'WN18RR': {'AMR': 0.853, '|M|': 18.813, 'Ties': 0.0},
                          'YAGO3-10': {'AMR': 0.977, '|M|': 14.077, 'Ties': 0.0}},
               'tucker': {'FB15K': {'AMR': 0.964, '|M|': 31.16, 'Ties': 0.352},
                          'FB15K237': {'AMR': 0.877, '|M|': 36.102, 'Ties': 0.274},
                          'WN18': {'AMR': 0.562, '|M|': 89.705, 'Ties': 0.749},
                          'WN18RR': {'AMR': 0.263, '|M|': 94.565, 'Ties': 0.353},
                          'YAGO3-10': {'AMR': 0.929, '|M|': 43.036, 'Ties': 0.0}}}

    mr2_data = {
        'boxe': {'BioKG': {'AMR': 0.978, '|M|': 9.948, 'Ties': 0}, 'Hetionet': {'AMR': 0.96, '|M|': 7.226, 'Ties': 0},
                 'NELL-995': {'AMR': 0.906, '|M|': 11.284, 'Ties': 0}},
        'complex': {'BioKG': {'AMR': 0.986, '|M|': 7.156, 'Ties': 0},
                    'Hetionet': {'AMR': 0.949, '|M|': 9.319, 'Ties': 0},
                    'NELL-995': {'AMR': 0.744, '|M|': 38.792, 'Ties': 0}},
        'hake': {'BioKG': {'AMR': 0.991, '|M|': 4.45, 'Ties': 0}, 'Hetionet': {'AMR': 0.963, '|M|': 6.336, 'Ties': 0},
                 'NELL-995': {'AMR': 0.654, '|M|': 45.916, 'Ties': 0}},
        'hole': {'BioKG': {'AMR': 0.986, '|M|': 7.28, 'Ties': 0}, 'Hetionet': {'AMR': 0.944, '|M|': 9.857, 'Ties': 0},
                 'NELL-995': {'AMR': 0.615, '|M|': 82.26, 'Ties': 0}},
        'quate': {'BioKG': {'AMR': 0.982, '|M|': 9.13, 'Ties': 0}, 'Hetionet': {'AMR': 0.944, '|M|': 10.733, 'Ties': 0},
                  'NELL-995': {'AMR': 0.663, '|M|': 64.846, 'Ties': 0}},
        'rotate': {'BioKG': {'AMR': 0.946, '|M|': 29.167, 'Ties': 0}, 'Hetionet': {'AMR': 0.96, '|M|': 6.78, 'Ties': 0},
                   'NELL-995': {'AMR': 0.344, '|M|': 163.952, 'Ties': 0}},
        'rotpro': {'BioKG': {'AMR': 0.897, '|M|': 55.333, 'Ties': 0},
                   'Hetionet': {'AMR': 0.962, '|M|': 6.454, 'Ties': 0},
                   'NELL-995': {'AMR': 0.546, '|M|': 116.101, 'Ties': 0}},
        'toruse': {'BioKG': {'AMR': 0.979, '|M|': 11.14, 'Ties': 0}, 'Hetionet': {'AMR': 0.953, '|M|': 8.47, 'Ties': 0},
                   'NELL-995': {'AMR': 0.755, '|M|': 55.159, 'Ties': 0}},
        'transe': {'BioKG': {'AMR': 0.992, '|M|': 3.956, 'Ties': 0},
                   'Hetionet': {'AMR': 0.955, '|M|': 7.733, 'Ties': 0},
                   'NELL-995': {'AMR': 0.819, '|M|': 27.616, 'Ties': 0}},
        'tucker': {'BioKG': {'AMR': 0.856, '|M|': 78.109, 'Ties': 0},
                   'Hetionet': {'AMR': 0.954, '|M|': 8.219, 'Ties': 0},
                   'NELL-995': {'AMR': 0.194, '|M|': 185.302, 'Ties': 0}}}

    data = {}
    for line in input:
        line = line.strip()

        if "n_mat_0" in line:
            dataset, model = line.split("Dataset: ")[1].split(" Model: ")[0], line.split("Model: ")[1].split(" ")[0]
            if dataset == "FB15K-237":
                dataset = dataset.replace("-", "")
            n_mat_0 = round(float(line.split("n_mat_0: ")[1].split(" ")[0]) / 1000000, 2)

            if model not in data:
                data[model] = {}

            if dataset not in data[model]:
                data[model][dataset] = {}

            if dataset == "Hetionet" or dataset == "BioKG" or dataset == "NELL-995":
                data[model][dataset] = {"AMR": str(round(mr2_data[model][dataset]["AMR"], 3)).replace("0.", "."),
                                        "|M|": str(n_mat_0)}
            else:
                data[model][dataset] = {"AMR": str(round(mr_data[model][dataset]["AMR"], 3)).replace("0.", "."),
                                        "|M|": str(n_mat_0)}

    table = create_multi_table(data, "Model name")

    latex = table.to_latex(column_format="|c|c|c|c|c|c|", index=False, index_names=False)

    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")
    print(latex)


def create_summary_is():
    input = open("input.txt")
    data = {}
    for line in input:
        # Check if the line contains dataset and model information
        if "Min:" in line and "Max:" in line and "Q1:" in line and "Q3:" in line and "Median:" in line:
            dataset, model = line.split("Dataset: ")[1].split(" Model: ")[0], line.split("Model: ")[1].split(" ")[0]
            # Create an empty dictionary for the dataset if it does not exist
            if model not in data:
                data[model] = {}
            # Create an empty dictionary for the model if it does not exist
            if dataset not in data[model]:
                data[model][dataset] = {}
            # Extract the data values from the line
            min_val = round(float(line.split("Min: ")[1].split(" ")[0]), 3)
            max_val = round(float(line.split("Max: ")[1].split(" ")[0]), 3)
            q1_val = round(float(line.split("Q1: ")[1].split(" ")[0]), 3)
            q3_val = round(float(line.split("Q3: ")[1].split(" ")[0]), 3)
            median_val = round(float(line.split("Median: ")[1]), 3)
            # Store the data values in the dictionary
            data[model][dataset]["Min"] = min_val
            data[model][dataset]["Max"] = max_val
            data[model][dataset]["Q1"] = q1_val
            data[model][dataset]["Q3"] = q3_val
            data[model][dataset]["Median"] = median_val

    table = create_multi_table(data, "Model name")
    latex = table.to_latex(index_names=False, index=False)
    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")

    latex = latex.replace("{l}", "{c||}").replace("0.", ".").replace("1.0", "1")
    print(latex)


if __name__ == "__main__":
    create_size_predictions_table()
    # create_summary_is()
    # create_box_plots()
    # check_for_rule_type_agreement_hetionet()
