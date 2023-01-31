from .tucker import TuckER
from .load_data import Data
import torch


class TuckERModelParser:
    def __init__(self, filename, dataset_name, model_name):
        self.filename = filename
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.dimensions = {"TuckER_FB15K_entity": 200, "TuckER_FB15K-237_entity": 200, "TuckER_WN18_entity": 200,
                           "TuckER_WN18RR_entity": 200, "TuckER_FB15K_relation": 200, "TuckER_FB15K-237_relation": 200,
                           "TuckER_WN18_relation": 30, "TuckER_WN18RR_relation": 30}

        self.hyperparameter = {"TuckER_FB15K_input_d": 0.2, "TuckER_FB15K-237_input_d": 0.3, "TuckER_WN18_input_d": 0.2,
                               "TuckER_WN18RR_input_d": 0.2, "TuckER_FB15K_hidden_d1": 0.2,
                               "TuckER_FB15K-237_hidden_d1": 0.4,
                               "TuckER_WN18_hidden_d1": 0.1, "TuckER_WN18RR_hidden_d1": 0.2,
                               "TuckER_FB15K_hidden_d2": 0.3,
                               "TuckER_FB15K-237_hidden_d2": 0.5, "TuckER_WN18_hidden_d2": 0.2,
                               "TuckER_WN18RR_hidden_d2": 0.3}

    def load_model(self, dataset):
        """
            Loads the kelpie model and then replaces embeddings with the trained embeddings

            Parameters:
                dataset (Dataset): The dataset object, used for getting num_entities and num_relations

            Returns:
                model (nn.module): The embedding algorithm required
        """

        model = TuckER(dataset, self.dimensions[self.model_name + "_" + self.dataset_name + "_entity"],
                       self.dimensions[self.model_name + "_" + self.dataset_name + "_relation"],
                       input_dropout=self.hyperparameter[self.model_name + "_" + self.dataset_name + "_input_d"],
                       hidden_dropout1=self.hyperparameter[self.model_name + "_" + self.dataset_name + "_hidden_d1"],
                       hidden_dropout2=self.hyperparameter[self.model_name + "_" + self.dataset_name + "_hidden_d2"])

        model.load_state_dict(torch.load(self.filename))
        return model

    def get_id_to_embedding(self):
        """
            Maps the entity and relationship names to their corresponding embeddings

            Returns:
                kelpie_entity_to_embedding (Dict): Dictionary containing entity names to embedding mapping
                kelpie_relation_to_embedding (Dict): Dictionary containing relation names to embedding mapping
        """
        dataset = Data("D:\PhD\Work\EmbeddingInterpretibility\TuckER\data\\" + self.dataset_name + "/", reverse=True)

        entity_idxs = {dataset.entities[i]: i for i in range(len(dataset.entities))}
        relation_idxs = {dataset.relations[i]: i for i in range(len(dataset.relations))}

        model = self.load_model(dataset)


        # We have model and dataset. Now we need to map kelpie_entity to embedding
        # Then we need to map our dataset entity to embedding

        tucker_entity_to_embedding = {}
        tucker_relation_to_embedding = {}

        for key, value in entity_idxs.items():
            tucker_entity_to_embedding[key] = model.E(torch.tensor([value]))[0].to("cpu").detach()

        for key, value in relation_idxs.items():
            tucker_relation_to_embedding[key] = model.R(torch.tensor([value]))[0].to("cpu").detach()

        W = model.W.data
        return tucker_entity_to_embedding, tucker_relation_to_embedding, W.to("cpu").detach(), model.bn0, model.bn1

    def map(self):
        """
            Maps the entity and relationship names from aKGE Datasets to their corresponding embeddings

            Returns:
                id_to_entity_embedding (Dict): Dictionary containing entity id to embedding mapping from aKGE datasets
                id_to_relation_embedding (Dict): Dictionary containing relation id to embedding mapping from aKGE datasets
        """

        tucker_entity_to_embedding, tucker_relation_to_embedding, W, bn0, bn1 = self.get_id_to_embedding()
        entity_file = open(
            "D:/PhD/Work/EmbeddingInterpretibility/AugmentedKGE/Datasets/" + self.dataset_name + "/entity2id.txt", "r")
        relation_file = open(
            "D:/PhD/Work/EmbeddingInterpretibility/AugmentedKGE/Datasets/" + self.dataset_name + "/relation2id.txt",
            "r")

        entity_file.readline()
        relation_file.readline()

        id_to_entity_embedding = {}
        id_to_relation_embedding = {}
        for line in entity_file:
            splits = line.strip().split("\t")
            entity = str(splits[0])
            id = int(splits[1])
            id_to_entity_embedding[id] = tucker_entity_to_embedding[entity]

        for line in relation_file:
            splits = line.strip().split("\t")
            relation = splits[0]
            id = int(splits[1])
            id_to_relation_embedding[id] = tucker_relation_to_embedding[relation]

        return id_to_entity_embedding, id_to_relation_embedding, W, bn0, bn1

