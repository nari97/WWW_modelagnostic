class InterpretibilityFileParser:

    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.positives_by_relation = {}
        self.positives_by_relation_train = {}
        self.positives_by_relation_test = {}
        self.positives_by_relation_valid = {}
        self.materialized_by_relation = {}
        self.mispredicted_by_relation = {}
        self.total_positives = 0
        self.total_materialized = 0
        self.total_mispredicted = 0

        self.count_positives_by_relation()
        self.count_materialized_by_relation()
        self.count_mispredicted_by_relation()

    def count_positives_by_relation(self):

        self.count_per_relation("../Datasets/" + self.dataset_name + "/train2id.txt", self.positives_by_relation_train)
        self.count_per_relation("../Datasets/" + self.dataset_name + "/test2id.txt", self.positives_by_relation_test)
        self.count_per_relation("../Datasets/" + self.dataset_name + "/valid2id.txt", self.positives_by_relation_valid)

        for pos_dict in [self.positives_by_relation_train, self.positives_by_relation_test,
                         self.positives_by_relation_valid]:
            #print (pos_dict)
            for key in pos_dict:
                if key not in self.positives_by_relation:
                    self.positives_by_relation[key] = 0

                self.positives_by_relation[key] += pos_dict[key]
                self.total_positives += pos_dict[key]


    def count_per_relation(self, filename, count_dict, type="r"):
        f_count = open(filename)
        if type == "r":
            f_count.readline()
        for line in f_count:
            if line == "\n":
                continue
            splits = line.strip().split("\t")
            if len(splits) == 1:
                splits = line.strip().split(" ")

            if type == "r":
                h, t, r = splits
            else:
                h, r, t = splits

            r = int(r)
            if r not in count_dict:
                count_dict[r] = 0

            count_dict[r] += 1


        f_count.close()
        return count_dict

    def count_materialized_by_relation(self):
        self.count_per_relation(
            "../Results/Materializations/" + self.dataset_name + "/" + self.model_name + "_materialized.tsv",
            self.materialized_by_relation, "t")

        for key in self.materialized_by_relation:
            self.total_materialized += self.materialized_by_relation[key]

    def count_mispredicted_by_relation(self):
        self.count_per_relation(
            "../Results/Materializations/" + self.dataset_name + "/" + self.model_name + "_mispredicted.tsv",
            self.mispredicted_by_relation, "t")

        for key in self.mispredicted_by_relation:
            self.total_mispredicted += self.mispredicted_by_relation[key]


if __name__ == "__main__":
    inter = InterpretibilityFileParser("TransE", "WN18RR")

    print(inter.positives_by_relation)
    print(inter.materialized_by_relation)
    print(inter.mispredicted_by_relation)

    print(inter.total_positives, inter.total_materialized, inter.total_mispredicted)
