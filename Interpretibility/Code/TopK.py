import os
from Interpretibility.Code.ParseRules import ParseRule, Atom, Rule
from GraphEditDistance import compute_graph_edit_distance


def compute_multi_rule_support(rules, gds):
    rule_queries = ["" for rule in rules]

    for i in range(0, len(rules)):
        rule = rules[i]

        for atom in rule.body_atoms:
            rule_queries[i] += " MATCH " + atom.neo4j_print()

        rule_queries[
            i] += " MATCH " + rule.head_atom.neo4j_print() + " WITH DISTINCT a,b RETURN a.entityId as headId, b.entityId as tailId"

    query = "CALL {"

    for rule_query in rule_queries:
        query += rule_query + " UNION "

    query = query[:-7] + "} RETURN count(*) as cnt"
    result = gds.run_cypher(query)

    return result.loc[0, "cnt"]


def compute_multi_rule_totals(rules, gds):
    if rules[0].functional_variable == "a":
        func = "b"
    else:
        func = "a"

    rule_queries = ["" for rule in rules]

    for i in range(0, len(rules)):
        rule = rules[i]

        for atom in rule.body_atoms:
            rule_queries[i] += " MATCH " + atom.neo4j_print()

        rule_queries[
            i] += " MATCH " + rule.head_atom.neo4j_print().replace(func,
                                                                   "") + " WITH DISTINCT a,b RETURN a.entityId as headId, b.entityId as tailId"

    query = "CALL {"

    for rule_query in rule_queries:
        query += rule_query + " UNION "

    query = query[:-7] + "} RETURN count(*) as cnt"
    result = gds.run_cypher(query)

    return result.loc[0, "cnt"]


def compute_multi_rule_pca(rules, support, gds):
    totals = compute_multi_rule_totals(rules, gds)
    return support * 1.0 / totals


def get_triples_for_relation(relation, gds):
    result = gds.run_cypher("MATCH ()-[r:`" + str(relation) + "`]->() RETURN count(*) as cnt")
    return result.loc[0, "cnt"]


def compute_combined_head_coverage(support, relation, gds):
    totals = get_triples_for_relation(relation, gds)
    return support * 1.0 / totals


def compute_selectivity(pca, hc, beta=1):
    selectivity = ((1 + beta * beta) * pca * hc) / (
            beta * beta * pca + hc)
    return selectivity


def compute_multi_rule_pca_internal(positives, totals):
    if len(totals) == 0:
        return 0

    return len(positives) * 1.0 / len(totals)


def compute_multi_rule_head_coverage_internal(positives, negative_count):
    return len(positives) * 1.0 / negative_count


def compute_combined_metrics_internal(positives, totals, negatives_count, beta=1):
    # print(negatives_count)
    pca = compute_multi_rule_pca_internal(positives, totals)
    hc = compute_multi_rule_head_coverage_internal(positives, negatives_count)
    selectivity = compute_selectivity(pca, hc, beta)

    return hc, pca, selectivity


def compute_combined_metrics(rules, gds, beta=1):
    support = compute_multi_rule_support(rules, gds)
    pca = compute_multi_rule_pca(rules, support, gds)
    hc = compute_combined_head_coverage(support, rules[0].head_atom.relationship, gds)
    selectivity = compute_selectivity(pca, hc, beta)

    return hc, pca, selectivity


def print_combined_rule(rules):
    result = ""

    for rule in rules:
        result += "( "
        for atom in rule.body_atoms:
            result += atom.relationship_print() + " "

        result += " ) V "

    result = result[:-3] + " ==> " + str(rules[0].head_atom.relationship_print())

    return result


def compute_overlap(rules, candidate, edit_distance_dict):
    min_overlap = 10000
    for rule in rules:
        key = (rule, candidate)
        overlap = edit_distance_dict[key]

        min_overlap = min(min_overlap, overlap)

    return min_overlap


def compute_metric(rules, candidate, edit_distance_dict):
    overlap = compute_overlap(rules, candidate, edit_distance_dict)
    pca = candidate.pca_confidence
    metric = overlap * pca
    return metric


def sort_rules_by_selectivity(rules):
    for rule in rules:
        rule.selectivity = compute_selectivity(rule.pca_confidence, rule.head_coverage, beta=2)

    rules = sorted(rules, key=lambda x: x.selectivity, reverse=True)
    return rules


def greedy_approximate(rules, negatives_count, edit_distance_dict, model_name, dataset_name, mat_type, rule_to_file):
    best_subset = [rules[0]]
    all_rules = rules.copy()

    rules = sort_rules_by_selectivity(rules)
    #     for rule in all_rules:
    #         print(rule.id_print(), rule.selectivity)

    head_predicate = str(rules[0].head_atom.relationship)
    if rules[0].pca_confidence >= 1.0 and rules[0].head_coverage >= 1.0:
        return print_combined_rule(best_subset), rules[0].head_coverage, rules[0].pca_confidence, rules[0].selectivity

    rules.remove(rules[0])
    while len(rules) > 0:
        best_rule = None
        best_metric = 0.0

        for rule in rules:
            metric = compute_metric(best_subset, rule, edit_distance_dict)

            if metric > best_metric:
                best_metric = metric
                best_rule = rule

        if best_rule is None:
            break

        best_subset.append(best_rule)
        rules.remove(best_rule)

        if len(best_subset) >= 5:
            break

    rule_positives = set()
    rule_totals = set()

    indexes = []

    ctr = 0

    for rule in best_subset:
        this_rule, positives, totals = get_params_from_rule_file(
            rule_to_file[rule.id_print()])
        if rule.id_print() != this_rule:
            print("Rule mismatch for:", str(
                all_rules.index(rule)), rule.id_print(), this_rule)
            exit(0)
        ctr += 1
        rule_positives = rule_positives.union(positives)
        rule_totals = rule_totals.union(totals)

    hc, pca, selectivity = compute_combined_metrics_internal(rule_positives, rule_totals, negatives_count, beta=2)

    return print_combined_rule(best_subset), hc, pca, selectivity


def rule_to_file_mapping(dataset_name, model_name, mat_type, predicate):
    folder = "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/" + str(
        predicate) + "/"
    rule_files = os.listdir(folder)

    rule_to_file = {}

    for rule_file in rule_files:
        f = open(folder + rule_file, "r")
        splits = f.readline().strip().split(",")
        rule = ",".join(splits[:-2])
        rule = rule.replace("=> ", "==>")
        rule = rule.replace("  ==>", " ==>")
        rule_to_file[rule] = folder + rule_file

    return rule_to_file


def best1(rules):
    best_rule = rules[0]
    hc = best_rule.head_coverage
    pca = best_rule.pca_confidence
    selectivity = compute_selectivity(pca, hc, beta=2)

    return print_combined_rule([best_rule]), hc, pca, selectivity


def best2(rules, negatives_count, dataset_name, model_name, mat_type):
    rule_1 = rules[0]
    rule_2 = rules[1]
    head_predicate = str(rules[0].head_atom.relationship)
    r1, positives_1, totals_1 = get_params_from_rule_file(
        "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/" + head_predicate + "/r" + str(
            0) + ".txt")
    r2, positives_2, totals_2 = get_params_from_rule_file(
        "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/" + head_predicate + "/r" + str(
            1) + ".txt")

    if r1 != rule_1.id_print():
        print("Rule mismatch:", rule_1.id_print(), r1)
        exit(0)

    if r2 != rule_2.id_print():
        print("Rule mismatch:", rule_2.id_print(), r2)
        exit(0)

    rule_positives = positives_1.union(positives_2)
    rule_totals = totals_1.union(totals_2)
    hc, pca, selec = compute_combined_metrics_internal(rule_positives, rule_totals, negatives_count, beta=2)
    return print_combined_rule([rule_1, rule_2]), hc, pca, selec


def best3(rules, negatives_count, dataset_name, model_name, mat_type):
    rule_1 = rules[0]
    rule_2 = rules[1]
    rule_3 = rules[2]
    head_predicate = str(rules[0].head_atom.relationship)
    r1, positives_1, totals_1 = get_params_from_rule_file(
        "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/" + head_predicate + "/r" + str(
            0) + ".txt")
    r2, positives_2, totals_2 = get_params_from_rule_file(
        "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/" + head_predicate + "/r" + str(
            1) + ".txt")
    r3, positives_3, totals_3 = get_params_from_rule_file(
        "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/" + head_predicate + "/r" + str(
            2) + ".txt")
    if r1 != rule_1.id_print():
        print("Rule mismatch:", rule_1.id_print(), r1)
        exit(0)

    if r2 != rule_2.id_print():
        print("Rule mismatch:", rule_2.id_print(), r2)
        exit(0)

    if r3 != rule_3.id_print():
        print("Rule mismatch:", rule_3.id_print(), r3)
        exit(0)
    rule_positives = positives_1.union(positives_2).union(
        positives_3)
    rule_totals = totals_1.union(totals_2).union(totals_3)

    hc, pca, selec = compute_combined_metrics_internal(rule_positives, rule_totals,
                                                       negatives_count, beta=2)
    return print_combined_rule([rule_1, rule_2, rule_3]), hc, pca, selec


def write_rule(rule, hc, pca, selec, f):
    f.write(rule + "\t" + str(round(hc, 4)) + "\t" + str(round(pca, 4)) + "\t" + str(round(selec, 4)) + "\n")


def count_negatives_by_predicate(model_name, dataset_name, mat_type):
    f = open("../Results/Materializations/" + dataset_name + "/" + model_name + "_" + mat_type + ".tsv")

    negative_dict = {}
    total = 0
    for line in f:
        if line == "\n":
            continue
        splits = line.strip().split("\t")
        if int(splits[1]) not in negative_dict:
            negative_dict[int(splits[1])] = 0
        total += 1
        negative_dict[int(splits[1])] += 1

    return negative_dict, total


def get_params_from_rule_file(filename):
    f = open(filename, "r")
    positives = set()
    totals = set()
    line = f.readline()
    splits = line.strip().split(",")
    rule = ",".join(splits[:-2])
    rule = rule.replace("=> ", "==>")
    rule = rule.replace("  ==>", " ==>")
    n_positives = int(f.readline().strip())

    for ctr in range(0, n_positives):
        line = f.readline().strip()
        h, t = line.split(",")
        inst = (h, t)
        positives.add(inst)

    f.readline()

    for line in f:
        line = line.strip()
        h, t = line.split(",")
        inst = (h, t)
        totals.add(inst)
    f.close()
    return rule, positives, totals


def run_greedy(dataset_name, model_name, mat_type):
    folder = "../Results/Instantiations/" + dataset_name + "_" + model_name + "_" + mat_type + "/"
    predicates = os.listdir(folder)
    f_best_1 = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best1.tsv",
        "w+")
    f_best_2 = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best2.tsv",
        "w+")
    #
    #     f_best_3 = open(
    #         "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best3.tsv",
    #         "w+")

    f_heuristic = open(
        "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\BestRules\\" + dataset_name + "\\" + model_name + "_" + mat_type + "_best_heuristic.tsv",
        "w+")

    negatives_count, negatives_totals = count_negatives_by_predicate(model_name, dataset_name, mat_type)

    count = len(predicates)

    mined_rules_folder = "../Results/MinedRules/" + dataset_name + "/" + model_name + "_" + mat_type + ".tsv"
    rp = ParseRule(mined_rules_folder, model_name, dataset_name, "\t")
    rp.parse_rules_from_file()
    i_ctr = 0
    for predicate in rp.rules_by_predicate:
        print("Predicate:", predicate, "Predicates left:", count - 1)
        rule_to_file = rule_to_file_mapping(dataset_name, model_name, mat_type, predicate)
        # print(rule_to_file)
        count -= 1
        totals_per_predicate = {}
        positives_per_predicate = {}
        rule_files = os.listdir(folder + str(predicate) + "/")
        print("\tInitializing positives and totals")
        for rule_file in rule_files:
            rule, positives, totals = get_params_from_rule_file(folder + str(predicate) + "/" + rule_file)
            totals_per_predicate[rule] = totals
            positives_per_predicate[rule] = positives

        # print(positives_per_predicate.keys())
        rules = rp.rules_by_predicate[predicate][:25]

        m1_rule, m1_hc, m1_pca, m1_selec = best1(rules.copy())
        write_rule(m1_rule, m1_hc, m1_pca, m1_selec, f_best_1)
        edit_distance_dict = compute_graph_edit_distance(rules)
        print("\t\tGED computed")

        try:
            m4_rule, m4_hc, m4_pca, m4_selec = greedy_approximate(rules.copy(), negatives_count[int(predicate)],
                                                                  edit_distance_dict, model_name=model_name,
                                                                  dataset_name=dataset_name, mat_type=mat_type,
                                                                  rule_to_file=rule_to_file)
            write_rule(m4_rule, m4_hc, m4_pca, m4_selec, f_heuristic)
            print("\t\tHeuristic completed")
        except:
            i_ctr += 1

        if len(rules) >= 2:
            m2_rule, m2_hc, m2_pca, m2_selec = best2(rules.copy(), negatives_count[int(predicate)],
                                                     model_name=model_name,
                                                     dataset_name=dataset_name, mat_type=mat_type)
            write_rule(m2_rule, m2_hc, m2_pca, m2_selec, f_best_2)
            print("\t\tBest 2 completed")
    #
    #         if len(rules) >= 3:
    #             m3_rule, m3_hc, m3_pca, m3_selec = best3(rules.copy(), negatives_count[int(predicate)],
    #                                                      model_name=model_name,
    #                                                      dataset_name=dataset_name, mat_type=mat_type)
    #             write_rule(m3_rule, m3_hc, m3_pca, m3_selec, f_best_3)
    #             print("\t\tBest 3 completed")

    print("Mismatch:", i_ctr)
    f_best_1.close()
    f_best_2.close()
    #     f_best_3.close()
    f_heuristic.close()


def run_experiment():
    models = ["TuckER"]
    dataset = ["WN18RR"]

    # f = open("../Results/Tables/" + dataset + "_IS_Results.csv", "w+")
    # f.write("Model,Best 1, Best 2,Best 3, Heuristic\n")
    for dataset_name in dataset:
        for model_name in models:
            print("Dataset:", dataset_name, " Model:", model_name)
            run_greedy(dataset_name=dataset_name, model_name=model_name, mat_type="materialized")
    # f.close()


if __name__ == "__main__":
    run_experiment()
