from ParseRules import ParseRule

if __name__ == "__main__":
    datasets = ["WN18", "WN18RR"]
    models = ["ConvE","TuckER"]

    path_to_rules = "D:\\PhD\\Work\\EmbeddingInterpretibility\\Interpretibility\\Results\\MinedRules\\"
    for model in models:
        for dataset in datasets:
            print (f"{model} {dataset}")
            boolean1 = False
            boolean2 = False
            path_to_materialized = f"{path_to_rules}{dataset}\\{model}_mispredicted.tsv"
            rp = ParseRule(filename=path_to_materialized, model_name=model, dataset_name=dataset)
            rp.parse_rules_from_file()

            for rule in rp.rules:
                if len(rule.body_atoms) == 1:
                    body = rule.body_atoms[0]
                    head = rule.head_atom

                    if body.relationship != head.relationship:

                        if body.variable1 == head.variable1 and not boolean1:
                            print(
                                f"Model:{model}, Dataset: {dataset}, Rule (X, p_1, Y) => (X, p_2, Y): {rule.relationship_print()}")
                            boolean1 = True

                        if body.variable1 == head.variable2 and not boolean2:
                            print(
                                f"Model:{model}, Dataset: {dataset}, Rule (X, p_1, Y) => (Y, p_2, X): {rule.relationship_print()}")
                            boolean2 = True

                    if boolean1 and boolean2:
                        break
