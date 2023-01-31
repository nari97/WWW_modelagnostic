import pandas as pd
def collect_materializations():

    datasets = ["FB15K", "FB15K-237", "WN18", "WN18RR"]
    models = ["ComplEx", "ConvE", "TransE", "TuckER"]

    folder = "../Results/Materializations/"
    for dataset in datasets:
        for model in models:
            num_mat = sum(1 for line in open(folder + dataset + "/" + model +"_materialized.tsv", "r"))
            num_mis = sum(1 for line in open(folder + dataset + "/" + model + "_mispredicted.tsv", "r"))

            print("Dataset:", dataset, "Model:", model, "n_mat:", num_mat, "n_mis:", num_mis)

def collect_pbe():
    models = ["TuckER"]
    datasets = ["WN18", "WN18RR", "FB15K", "FB15K-237"]

    folder_to_dataset = "D:\\PhD\\Work\\EmbeddingInterpretibility\\Interpretibility\\Datasets\\"
    folder_to_pbe = "D:\\\PhD\\\Work\\\EmbeddingInterpretibility\\\Interpretibility\\\Results\\Materializations\\"
    for model in models:
        for dataset in datasets:
            test_file_path = f"{folder_to_dataset}{dataset}\\test2id.txt"
            pbe_file_path = f"{folder_to_pbe}{dataset}\\{model}_positives_before_expected.tsv"
            n_test_triples = int(open(test_file_path).readline())

            n_pbe = 0
            with open(pbe_file_path) as f:
                for line in f:
                    splits = line.strip().split("\t")
                    n_pbe += int(splits[1])

            print(f"Model: {model}, Dataset: {dataset}, Misp: {((2 * n_test_triples) - n_pbe) / (2 * n_test_triples)}")

if __name__ == "__main__":
    collect_materializations()
    collect_pbe()
