from ModelMaterializer import materialize_individual
import os
if __name__ == "__main__":
    model_name = "ComplEx"
    dataset_name = "FB15K-237"
    print ("Running materializer")
    materialization_filename = "..\Results\Materializations\\" + dataset_name + "\\" + model_name +"_materialized.tsv"
    materialize_individual(model_name=model_name, dataset_name=dataset_name, use_gpu=True)
    materialize_rule_filename = "..\Results\MinedRules\\" + dataset_name + "\\" + model_name +"_materialized.tsv"

    print ("Running rule miner")
    os_subprocess_call = "java -Xmx8g -jar \"..\\amie-dev.jar\" \"" + materialization_filename + "\"  --datalog --nc 4 -mins 1 -minis 1 > \"" + materialize_rule_filename + "\""
    print (os_subprocess_call)
    os.system(os_subprocess_call)

    misprediction_filename = "..\Results\Materializations\\" + dataset_name + "\\" + model_name + "_mispredicted.tsv"
    misprediction_rule_filename = "..\Results\MinedRules\\" + dataset_name + "\\" + model_name +"_mispredicted.tsv"
    os_subprocess_call = "java -Xmx24g -jar \"..\\amie-dev.jar\" \"" + misprediction_filename + "\"  --datalog --nc 4 -mins 1 -minis 1 > \"" + misprediction_rule_filename + "\""
    print(os_subprocess_call)
    os.system(os_subprocess_call)