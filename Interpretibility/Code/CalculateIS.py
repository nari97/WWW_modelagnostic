from ParseRules import ParseRule
from InterpretibilityFileParser import InterpretibilityFileParser


def get_relation_to_id(dataset_name, delimiter="\t"):
    folder = "../Datasets/" + dataset_name + "/"

    f = open(folder + "relation2id.txt")
    relation_to_id = {}

    f.readline()

    for line in f:
        splits = line.strip().split(delimiter)

        relation_to_id[splits[0]] = int(splits[1])

    return relation_to_id


def get_test_triple_count(dataset_name, delimiter=" "):
    folder = "../Datasets/" + dataset_name + "/"

    f = open(folder + "test2id.txt")
    test_triple_count_predicate = {}
    test_triple_count_total = 0

    f.readline()

    for line in f:
        splits = line.strip().split(delimiter)

        if int(splits[2]) not in test_triple_count_predicate:
            test_triple_count_predicate[int(splits[2])] = 0

        test_triple_count_predicate[int(splits[2])] += 1
        test_triple_count_total += 1

    return test_triple_count_predicate, test_triple_count_total


def get_negative_triple_count(dataset_name, model_name, mat_type, delimiter="\t"):
    folder = "../Results/Materializations/" + dataset_name + "/" + model_name + "_" + mat_type+".tsv"

    f = open(folder)
    test_triple_count_predicate = {}
    test_triple_count_total = 0

    f.readline()

    for line in f:
        if line == "\n":
            continue
        splits = line.strip().split(delimiter)

        if int(splits[1]) not in test_triple_count_predicate:
            test_triple_count_predicate[int(splits[1])] = 0

        test_triple_count_predicate[int(splits[1])] += 1
        test_triple_count_total += 1

    return test_triple_count_predicate, test_triple_count_total

def get_is(f_rule, relation_to_id, test_triple_count_predicate, test_triple_count_total):
    score = 0.0

    for line in f_rule:
        splits = line.strip().split("\t")
        selec = float(splits[-1])
        head = splits[0].split(" ==> ")[1].replace("(?a,?b)", "")
        head_id = relation_to_id[head]
        score += test_triple_count_predicate[head_id] * selec

    return round(score * 1.0 / test_triple_count_total,4)

def get_is_hc_pca(f_rule, relation_to_id, test_triple_count_predicate, test_triple_count_total):
    score_selec = 0.0
    score_hc = 0.0
    score_pca = 0.0

    for line in f_rule:
        splits = line.strip().split("\t")
        selec = float(splits[-1])
        pca = float(splits[-2])
        hc = float(splits[-3])
        head = splits[0].split(" ==> ")[1].replace("(?a,?b)", "")
        head_id = relation_to_id[head]
        score_selec += test_triple_count_predicate[head_id] * selec
        score_hc += test_triple_count_predicate[head_id] * hc
        score_pca += test_triple_count_predicate[head_id] * pca

    return [round(score_selec * 1.0 / test_triple_count_total, 4), round(score_hc * 1.0 / test_triple_count_total, 4), round(score_pca * 1.0 / test_triple_count_total, 4)]

def run_individual(dataset_name, model_name, mat_type, relation_to_id, test_triple_count_predicate,
                   test_triple_count_total):
    folder = "../Results/BestRules/" + dataset_name + "/"

    f_best1 = open(folder + model_name + "_" + mat_type + "_best1.tsv", "r")
    f_best2 = open(folder + model_name + "_" + mat_type + "_best2.tsv", "r")
    f_best3 = open(folder + model_name + "_" + mat_type + "_best3.tsv", "r")
    f_best_h = open(folder + model_name + "_" + mat_type + "_best_heuristic.tsv", "r")

    best1_is = get_is_hc_pca(f_best1, relation_to_id, test_triple_count_predicate, test_triple_count_total)
    best2_is = get_is_hc_pca(f_best2, relation_to_id, test_triple_count_predicate, test_triple_count_total)
    best3_is = get_is_hc_pca(f_best3, relation_to_id, test_triple_count_predicate, test_triple_count_total)
    heuristic_is = get_is_hc_pca(f_best_h, relation_to_id, test_triple_count_predicate, test_triple_count_total)

    f_best1.close()
    f_best2.close()
    f_best3.close()
    f_best_h.close()

    return best1_is, best2_is, best3_is, heuristic_is


def calculate_is():
    datasets = ["WN18RR", "FB15K-237", "WN18"]
    models = ["ComplEx","ConvE","TransE","TuckER"]
    mat_type = "materialized"
    for dataset_name in datasets:
        relation_to_id = get_relation_to_id(dataset_name)
        test_triple_count_predicate, test_triple_count_total = get_test_triple_count(dataset_name)
        #print(test_triple_count_predicate)
        f_result = open("../Results/Tables/" + dataset_name + "_" + mat_type + "_final_result" + ".tsv", "w+")
        f_result.write("Model,Best 1,Best 2,Best 3,Heuristic\n")
        for model_name in models:
            #test_triple_count_predicate, test_triple_count_total = get_negative_triple_count(dataset_name, model_name,
            #                                                                                 mat_type)
            best1, best2, best3, best_h = run_individual(dataset_name=dataset_name, model_name=model_name, relation_to_id=relation_to_id, mat_type=mat_type,
                                                         test_triple_count_predicate=test_triple_count_predicate, test_triple_count_total=test_triple_count_total)
            f_result.write(model_name + "," + str(best1) + "," + str(best2) + "," + str(best3) + "," + str(best_h) + "\n")

        f_result.close()

def calculate_aggregates():
    datasets = ["WN18RR"]
    models = ["TuckER"]
    mat_type = "materialized"
    for dataset_name in datasets:
        relation_to_id = get_relation_to_id(dataset_name)
        test_triple_count_predicate, test_triple_count_total = get_test_triple_count(dataset_name)
        # print(test_triple_count_predicate)
        f_result_selec = open("../Results/Tables/" + dataset_name + "_" + mat_type + "_final_result_selec" + ".tsv", "w+")
        f_result_selec.write("Model,Best 1,Best 2,Best 3,Heuristic\n")

        f_result_hc = open("../Results/Tables/" + dataset_name + "_" + mat_type + "_final_result_hc" + ".tsv", "w+")
        f_result_hc.write("Model,Best 1,Best 2,Best 3,Heuristic\n")

        f_result_pca = open("../Results/Tables/" + dataset_name + "_" + mat_type + "_final_result_pca" + ".tsv", "w+")
        f_result_pca.write("Model,Best 1,Best 2,Best 3,Heuristic\n")

        for model_name in models:
            # test_triple_count_predicate, test_triple_count_total = get_negative_triple_count(dataset_name, model_name,
            #                                                                                 mat_type)
            best1, best2, best3, best_h = run_individual(dataset_name=dataset_name, model_name=model_name,
                                                         relation_to_id=relation_to_id, mat_type=mat_type,
                                                         test_triple_count_predicate=test_triple_count_predicate,
                                                         test_triple_count_total=test_triple_count_total)
            f_result_selec.write(
                model_name + "," + str(best1[0]) + ","
                + str(best2[0])+ "," + str(best3[0])+ "," + "," + str(best_h[0]) + "\n")
            f_result_hc.write(
                model_name + "," + str(best1[1]) + ","
                + str(best2[1])+ "," + str(best3[1])+ "," + "," + str(best_h[1]) + "\n")

            f_result_pca.write(
                model_name + "," + str(best1[2]) + ","
                + str(best2[2]) + "," + str(best3[2]) + "," + "," + str(best_h[2]) + "\n")

        f_result_selec.close()
        f_result_hc.close()
        f_result_pca.close()

if __name__ == "__main__":
    calculate_aggregates()
