from Code import ParseRules
from Code.LatexUtils import create_multi_table


def get_entity_types(relationship):
    index = 0
    for i in range(len(relationship)):
        val = relationship[i]
        if 97 <= ord(val) <= 122:
            index = i
            break

    if ">" in relationship:
        lhs = relationship[0:index]
        rhs = relationship[index + 2:]
    else:
        lhs = relationship[0:index]
        rhs = relationship[index + 1:]

    return [lhs, rhs]


def check_rules(rules, entity_types):
    match = 0

    for rule in rules:
        type_dict = {}
        atoms = list(rule.body_atoms)
        atoms.append(rule.head_atom)
        for atom in atoms:
            relationship = atom.relationship_name

            lhs, rhs = entity_types[relationship]

            if atom.variable1 not in type_dict:
                type_dict[atom.variable1] = set()

            type_dict[atom.variable1].add(lhs)

            if atom.variable2 not in type_dict:
                type_dict[atom.variable2] = set()

            type_dict[atom.variable2].add(rhs)

        flag = True
        for key in type_dict:
            if len(type_dict[key]) > 1:
                flag = False
                break

        if flag:
            match += 1

    return match


def check_rule_overlap(rules, file_to_write):
    rule_dict = {}

    for model_name in rules:
        rules_by_model = rules[model_name]

        for rule in rules_by_model:
            rp = rule.relationship_print()

            if rp not in rule_dict:
                rule_dict[rp] = set()

            rule_dict[rp].add((model_name, rule.selectivity))

    predicates = {}
    rule_stats = {}
    for rule in rule_dict:
        head_predicate = rule.split("==>")[1].split("(")[0].strip()

        if head_predicate not in predicates:
            predicates[head_predicate] = 0

        longest_so_far = predicates[head_predicate]

        if len(rule_dict[rule]) >= longest_so_far:
            predicates[head_predicate] = len(rule_dict[rule])

            rule_stats[head_predicate] = {"Rule": rule, "Models": rule_dict[rule]}

    predicates_sorted = sort_dict_by_value_desc(predicates)

    for val in predicates_sorted:
        rule = rule_stats[val[0]]["Rule"]
        models_selec = rule_stats[val[0]]["Models"]
        file_to_write.write(f"Predicate: {val[0]}; Rule: {rule}; Count: {val[1]}; Model stats: {models_selec}\n")


def sort_dict_by_value_desc(d):
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [[k, v] for k, v in sorted_items]


def get_hetionet_relation_types():
    hetionet_folder = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets\Hetionet"

    entity_types = {}
    with open(f"{hetionet_folder}/relation2id.txt") as f:
        f.readline()
        for line in f:
            relation = line.split("\t")[0]
            entity_types[relation] = get_entity_types(relation)

    return entity_types


def get_biokg_relation_types():
    biokg_folder = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets\BioKG"

    all_entities = {}
    all_triples = {}
    for type_of_file in ["metadata.disease", "metadata.drug", "metadata.pathway", "metadata.protein",
                         "properties.genetic_disorder"]:
        with open(f"{biokg_folder}/biokg.{type_of_file}.tsv") as f:
            for line in f:
                line = line.strip()
                entity = line.split("\t")[0]

                all_entities[entity] = type_of_file.split(".")[1]

    with open(f"{biokg_folder}/biokg.properties.pathway.tsv") as f:
        for line in f:
            line = line.strip()
            entity1 = line.split("\t")[0]
            entity2 = line.split("\t")[2]

            all_entities[entity1] = "pathway"
            all_entities[entity2] = "pathway"

    with open(f"{biokg_folder}/biokg.links.tsv") as f:
        for line in f:
            line = line.strip()
            head, relation, tail = line.split("\t")

            if relation not in all_triples:
                all_triples[relation] = []
            all_triples[relation].append([head, relation, tail])

    entity_types = {}
    for relation in all_triples:
        lhs = set()
        rhs = set()

        for triple in all_triples[relation]:
            try:

                if triple[0] not in all_entities:
                    if "R-" in triple[0]:
                        lhs.add("pathway")
                    elif "MIM" in triple[0]:
                        lhs.add("genetic_disorder")
                else:
                    lhs.add(all_entities[triple[0]])

                if triple[2] not in all_entities:
                    if "R-" in triple[2]:
                        rhs.add("pathway")
                    elif "MIM" in triple[2]:
                        rhs.add("genetic_disorder")
                else:
                    rhs.add(all_entities[triple[2]])
            except:
                print(f"{relation},{triple}")

        entity_types[relation] = [lhs.pop(), rhs.pop()]

    return entity_types


def check_for_rule_type_agreement(dataset_name):
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    folder_to_datasets = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets"
    folder_to_mined_rules = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules"

    if dataset_name == "Hetionet":
        entity_types = get_hetionet_relation_types()
    else:
        entity_types = get_biokg_relation_types()

    rules_by_model = {}

    result = {}
    for model in models:
        rp = ParseRules.ParseRule(
            filename=f"{folder_to_mined_rules}/{dataset_name}/{model}/{dataset_name}_{model}_mat_0_rules.tsv",
            model_name=model,
            dataset_name=dataset_name, folder_to_datasets=folder_to_datasets)
        rp.parse_rules_from_file(beta=1.0)
        rules = rp.rules
        rules_by_model[model] = rules
        match = check_rules(rules, entity_types)

        result[model] = {dataset_name: {"Number of rules mined": len(rules),
                                        "Ratio of type agreeing rules:": round(match * 1.0 / len(rules), 3)}}

    return result


def check_for_rule_type_agreement_bio_datasets_experiment():
    result_het = check_for_rule_type_agreement("Hetionet")
    result_biokg = check_for_rule_type_agreement("BioKG")

    data = {}

    models = result_het.keys()

    for model_name in models:
        data[model_name] = {"Hetionet": result_het[model_name]["Hetionet"], "BioKG": result_biokg[model_name]["BioKG"]}

    table = create_multi_table(data, "Model name")
    latex = table.to_latex(index_names=False, index=False)
    latex = latex.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "").replace("\n",
                                                                                                        " \\hline \n")

    latex = latex.replace("{l}", "{c||}").replace("0.", ".").replace("1.0", "1")
    # print(table.to_markdown())
    print(latex)
    print(table)


def count_rules_by_model(dataset_name):
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    folder_to_datasets = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets"
    folder_to_mined_rules = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules"

    rules_by_model = {}

    f = open(
        fr"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\Tables\RuleOverlapAcrossModels/{dataset_name}_overlap.txt",
        "w+")
    for model in models:
        rp = ParseRules.ParseRule(
            filename=f"{folder_to_mined_rules}/{dataset_name}/{model}/{dataset_name}_{model}_mat_0_rules.tsv",
            model_name=model,
            dataset_name=dataset_name, folder_to_datasets=folder_to_datasets)
        rp.parse_rules_from_file(beta=1.0)
        rules = rp.rules
        rules_by_model[model] = rules

    check_rule_overlap(rules_by_model, f)
    f.close()


def same_dataset_rule_comparison(dataset1, dataset2):
    models = ["boxe", "complex", "hake", "hole", "quate", "rotate", "rotpro", "toruse", "transe", "tucker"]
    folder_to_datasets = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Datasets"
    folder_to_mined_rules = r"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules"

    rules_by_dataset = {}
    for dataset_name in [dataset1, dataset2]:
        for model in models:
            rp = ParseRules.ParseRule(
                filename=f"{folder_to_mined_rules}/{dataset_name}/{model}/{dataset_name}_{model}_mat_0_rules.tsv",
                model_name=model,
                dataset_name=dataset_name, folder_to_datasets=folder_to_datasets)
            rp.parse_rules_from_file(beta=1.0)
            rules = rp.rules

            for rule in rules:
                rp = rule.relationship_print()

                if rp not in rules_by_dataset:
                    rules_by_dataset[rp] = {}

                if dataset_name not in rules_by_dataset[rp]:
                    rules_by_dataset[rp][dataset_name] = {}

                rules_by_dataset[rp][dataset_name][model] = rule.selectivity

    filtered_rules_appearing_in_both = {}

    rules_by_dataset_model = {}

    for rule in rules_by_dataset:
        for dataset_name in rules_by_dataset[rule]:
            if rule not in rules_by_dataset_model:
                rules_by_dataset_model[rule] = {}

            rules_by_dataset_model[rule][dataset_name] = set(rules_by_dataset[rule][dataset_name].keys())

    for rule in rules_by_dataset_model:

        if dataset1 in rules_by_dataset_model[rule] and len(rules_by_dataset_model[rule][dataset1]) > 0 and dataset2 in \
                rules_by_dataset_model[rule] and len(rules_by_dataset_model[rule][dataset2]) > 0:
            filtered_rules_appearing_in_both[rule] = rules_by_dataset_model[rule]

    filtered_rules_count_by_model = {}

    for rule in filtered_rules_appearing_in_both:
        matching_models = filtered_rules_appearing_in_both[rule][dataset1].intersection(
            filtered_rules_appearing_in_both[rule][dataset2])

        if len(matching_models)>0:
            filtered_rules_count_by_model[rule] = len(matching_models)

    filtered_rules_count_by_model_sorted = sort_dict_by_value_desc(filtered_rules_count_by_model)

    f = open(fr"D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\Tables\RuleOverlapAcrossDatasets/{dataset1[0:2]}_overlap.txt", "w+")
    for val in filtered_rules_count_by_model_sorted:
        rule = val[0]
        count = val[1]
        matching_models = filtered_rules_appearing_in_both[rule][dataset1].intersection(
            filtered_rules_appearing_in_both[rule][dataset2])

        matching_model_dict = {}
        for model in matching_models:
            if model not in matching_model_dict:
                matching_model_dict[model] = {}

            matching_model_dict[model] = {dataset1: rules_by_dataset[rule][dataset1][model],
                                          dataset2: rules_by_dataset[rule][dataset2][model]}

        f.write(f"Rule: {rule}; Count: {count}; Models: {matching_model_dict}\n")
    f.close()

def same_dataset_rule_comparison_experiment():
    same_dataset_rule_comparison("WN18", "WN18RR")
    same_dataset_rule_comparison("FB15K", "FB15K-237")


def count_rules_experiment():
    datasets = ["FB15K", "FB15K-237", "WN18", "WN18RR", "NELL-995", "YAGO3-10", "Hetionet", "BioKG"]

    for dataset in datasets:
        count_rules_by_model(dataset)


if __name__ == "__main__":
    # check_for_rule_type_agreement_bio_datasets()
    # count_rules_experiment()
    same_dataset_rule_comparison_experiment()
