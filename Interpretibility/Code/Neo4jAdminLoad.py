def create_relationship_file(model_name, dataset_name, mat_type="materialized"):
    f_read = open("../Results/Materializations/" + dataset_name + "/" + model_name + "_" + mat_type + ".tsv", "r")
    f_write = open("../Results/Materializations/" + dataset_name + "/" + model_name + "_" + mat_type + "_neo4j.csv",
                   "w+")

    f_write.write(":START_ID,:TYPE,:END_ID\n")

    for line in f_read:
        if line == "\n":
            continue
        splits = line.strip().split("\t")
        f_write.write(splits[0] + "," + splits[1] + "," + splits[2] + "\n")

    f_read.close()
    f_write.close()


if __name__ == "__main__":
    mat_type = "materialized"
    create_relationship_file("TuckER", "WN18", mat_type)
    create_relationship_file("TuckER", "WN18RR", mat_type)
    create_relationship_file("TuckER", "WN18", "mispredicted")
    create_relationship_file("TuckER", "WN18RR", "mispredicted")
    create_relationship_file("TuckER", "FB15K", mat_type)
    create_relationship_file("TuckER", "FB15K-237", mat_type)
