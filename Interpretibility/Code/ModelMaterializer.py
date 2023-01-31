import os

from AugmentedKGE.DataLoader.TripleManager import TripleManager
import torch
import time
import sys
import glob
import math

from AugmentedKGE.Train.Materializer import Materializer
from AugmentedKGE.Train.GraphMLMaterializer import GraphMLMaterializer
from Kelpie.KelpieModelParser import KelpieModelParser
from TuckER.TuckERModelParser import TuckERModelParser
from Kelpie.aKGEImplementations.KelpieTransE import KelpieTransE
from Kelpie.aKGEImplementations.KelpieComplEx import KelpieComplEx
from Kelpie.aKGEImplementations.KelpieConvE import KelpieConvE
from Kelpie.aKGEImplementations.KelpieTuckER import KelpieTuckER
import networkx


def get_model(model_name, dataset_name, ent_total, rel_total, dim=0, dim_e=0, dim_r=0):
    if model_name == "TransE":
        model = KelpieTransE(ent_total, rel_total, dim=dim_e)
        neg_predict = True

    elif model_name == "ComplEx":
        model = KelpieComplEx(ent_total, rel_total, dim=int(dim_e / 2))
        neg_predict = True

    elif model_name == "ConvE":
        model = KelpieConvE(ent_total, rel_total, dim=dim_e)
        neg_predict = True

    elif model_name == "TuckER":
        hyperparameter = {"TuckER_FB15K_input_d": 0.2, "TuckER_FB15K-237_input_d": 0.3, "TuckER_WN18_input_d": 0.2,
                          "TuckER_WN18RR_input_d": 0.2, "TuckER_FB15K_hidden_d1": 0.2,
                          "TuckER_FB15K-237_hidden_d1": 0.4,
                          "TuckER_WN18_hidden_d1": 0.1, "TuckER_WN18RR_hidden_d1": 0.2,
                          "TuckER_FB15K_hidden_d2": 0.3,
                          "TuckER_FB15K-237_hidden_d2": 0.5, "TuckER_WN18_hidden_d2": 0.2,
                          "TuckER_WN18RR_hidden_d2": 0.3}
        model = KelpieTuckER(ent_total, rel_total, dim_e=dim_e, dim_r=dim_r,
                             input_d=hyperparameter[model_name + "_" + dataset_name + "_input_d"],
                             hidden_d1=hyperparameter[model_name + "_" + dataset_name + "_hidden_d1"],
                             hidden_d2=hyperparameter[model_name + "_" + dataset_name + "_hidden_d2"])
        neg_predict = True

    return model, neg_predict


def materialize_individual(dataset_name, model_name, use_gpu=False):
    if model_name == "TuckER":
        trained_model_filename = "../../TuckER/tucker_trained_models/" + model_name + "_" + dataset_name + ".pt"
    else:
        trained_model_filename = "../../Kelpie/kelpie_trained_models/" + model_name + "_" + dataset_name + ".pt"

    print("Loading Model")
    start = time.perf_counter()

    if model_name == "TuckER":
        model = TuckERModelParser(trained_model_filename, dataset_name, model_name)
    else:
        model = KelpieModelParser(trained_model_filename, dataset_name=dataset_name, model_name=model_name)

    if model_name == "TuckER":
        e, r, W, bn0, bn1 = model.map()
    else:
        e, r = model.map()

    ent_total = len(e)
    rel_total = len(r)
    dim_e = e[0].shape[0]
    dim_r = r[0].shape[0]
    akge_model, neg_predict = get_model(model_name=model_name, ent_total=ent_total, rel_total=rel_total, dim_e=dim_e,
                                        dim_r=dim_r, dataset_name=dataset_name)
    akge_model.initialize_model()
    print(f"Entity shape: {ent_total} {dim_e}")
    print(f"Relation shape: {rel_total} {dim_r}")
    if model_name == "TuckER":
        akge_model.load_kelpie_model(e, r, W, bn0, bn1)
    else:
        akge_model.load_kelpie_model(e, r)

    end = time.perf_counter()

    if use_gpu:
        akge_model.set_use_gpu(use_gpu=use_gpu)
        akge_model.bn0 = akge_model.bn0.to("cuda:0")
        akge_model.bn1 = akge_model.bn1.to("cuda:0")

    print("Time elapsed to load model:", str(end - start))

    corruption_mode = "LCWA"

    path = "../../Interpretibility/Datasets/" + dataset_name + "/"

    split_prefix = ""
    print("Loading Triple Manager")
    start = time.perf_counter()
    manager = TripleManager(path, splits=[split_prefix + "test", split_prefix + "valid", split_prefix + "train"],
                            corruption_mode=corruption_mode)
    end = time.perf_counter()

    print("Time elapsed to load TripleManager:", str(end - start))

    results_folder_path = "D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\Materializations\\" + dataset_name
    print("Starting Materializer")
    start = time.perf_counter()
    materializer = GraphMLMaterializer(manager, use_gpu=use_gpu, neg_predict=neg_predict)

    collector, networkx_graph = materializer.materialize(model=akge_model, name=results_folder_path)

    if not os.path.exists(f"{results_folder_path}//{dataset_name.lower()}_{model_name.lower()}_graphml.graphml"):
        open(f"{results_folder_path}//{dataset_name.lower()}_{model_name.lower()}_graphml.graphml", 'w').close()
    networkx.write_graphml(networkx_graph,
                           f"{results_folder_path}//{dataset_name.lower()}_{model_name.lower()}_graphml.graphml",
                           prettyprint=True)

    adjusted_mr = 1.0 - collector.get_metric(metric_str="mr").get() * 1.0 / collector.get_expected(
        metric_str="mr").get()
    adjusted_mrh = 1.0 - collector.get_metric(metric_str="mrh").get() * 1.0 / collector.get_expected(
        metric_str="mrh").get()
    adjusted_mrg = 1.0 - collector.get_metric(metric_str="mrg").get() * 1.0 / collector.get_expected(
        metric_str="mrg").get()

    print(f"Adjusted MR: {adjusted_mr}\tAdjusted MRH: {adjusted_mrh}\tAdjusted MRG: {adjusted_mrg}")
    end = time.perf_counter()
    print("Time elapsed to materialize:", str(end - start))


if __name__ == '__main__':
    materialize_individual(model_name="TransE", dataset_name="WN18RR", use_gpu=False)
