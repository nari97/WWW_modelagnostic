from Kelpie.link_prediction.models.transe import TransE
from Kelpie.link_prediction.models.complex import ComplEx
from Kelpie.link_prediction.models.conve import ConvE
from Kelpie.dataset import Dataset
from Kelpie.link_prediction.models.tucker import TuckER
from Kelpie.link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, \
    REGULARIZER_WEIGHT, INPUT_DROPOUT, DECAY, LABEL_SMOOTHING, \
    FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE, INIT_SCALE
import torch
import pickle


class KelpieModelParser:
    def __init__(self, filename, dataset_name, model_name):
        self.filename = filename
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.dimensions = {"TransE_FB15k": 200, "ComplEx_FB15K": 2000, "ConvE_FB15K": 200, "TransE_FB15K-237": 50,
                           "ComplEx_FB15K-237": 1000, "ConvE_FB15K-237": 200, "TransE_WN18": 50, "ComplEx_WN18": 500,
                           "ConvE_WN18": 200, "TransE_WN18RR": 50, "ComplEx_WN18RR": 500, "ConvE_WN18RR": 200,
                           "TransE_YAGO3-10": 200, "ComplEx_YAGO3-10": 1000, "ConvE_YAGO3-10": 200}

        self.hyperparameters = {DIMENSION: self.dimensions[model_name + "_" + dataset_name],
                                MARGIN: 2.0,
                                NEGATIVE_SAMPLES_RATIO: 5,
                                REGULARIZER_WEIGHT: 500,
                                BATCH_SIZE: 2048,
                                LEARNING_RATE: 0.0004,
                                EPOCHS: 250,
                                INIT_SCALE: 1e-3,
                                DECAY: 0.995,
                                INPUT_DROPOUT: 0.3,
                                FEATURE_MAP_DROPOUT: 0.5,
                                LABEL_SMOOTHING: 0.1,
                                HIDDEN_LAYER_SIZE: 9728,
                                HIDDEN_DROPOUT: 0.1
                                }

    def load_model(self, dataset):
        """
            Loads the kelpie model and then replaces embeddings with the trained embeddings

            Parameters:
                dataset (Dataset): The dataset object, used for getting num_entities and num_relations

            Returns:
                model (nn.module): The embedding algorithm required
        """

        if self.model_name == "TransE":
            model = TransE(dataset=dataset, hyperparameters=self.hyperparameters, init_random=True)
        elif self.model_name == "ComplEx":
            model = ComplEx(dataset=dataset, hyperparameters=self.hyperparameters, init_random=True)
        elif self.model_name == "ConvE":
            model = ConvE(dataset=dataset, hyperparameters=self.hyperparameters, init_random=True, )
        model.load_state_dict(torch.load(self.filename))
        return model

    def get_id_to_embedding(self):
        """
            Maps the entity and relationship names to their corresponding embeddings

            Returns:
                kelpie_entity_to_embedding (Dict): Dictionary containing entity names to embedding mapping
                kelpie_relation_to_embedding (Dict): Dictionary containing relation names to embedding mapping
        """
        dataset = Dataset(self.dataset_name)

        model = self.load_model(dataset)

        # We have model and dataset. Now we need to map kelpie_entity to embedding
        # Then we need to map our dataset entity to embedding

        kelpie_entity_to_embedding = {}
        kelpie_relation_to_embedding = {}

        for key, value in dataset.entity_name_2_id.items():
            kelpie_entity_to_embedding[key] = model.entity_embeddings[value, :].to("cpu").detach()

        for key, value in dataset.relation_name_2_id.items():
            kelpie_relation_to_embedding[key] = model.relation_embeddings[value, :].to("cpu").detach()

        return kelpie_entity_to_embedding, kelpie_relation_to_embedding

    def map(self):
        """
            Maps the entity and relationship names from aKGE Datasets to their corresponding embeddings

            Returns:
                id_to_entity_embedding (Dict): Dictionary containing entity id to embedding mapping from aKGE datasets
                id_to_relation_embedding (Dict): Dictionary containing relation id to embedding mapping from aKGE datasets
        """

        kelpie_entity_to_embedding, kelpie_relation_to_embedding = self.get_id_to_embedding()
        entity_file = open("D:/PhD/Work/EmbeddingInterpretibility/AugmentedKGE/Datasets/" + self.dataset_name + "/entity2id.txt", "r")
        relation_file = open("D:/PhD/Work/EmbeddingInterpretibility/AugmentedKGE/Datasets/" + self.dataset_name + "/relation2id.txt", "r")

        entity_file.readline()
        relation_file.readline()

        id_to_entity_embedding = {}
        id_to_relation_embedding = {}
        for line in entity_file:
            splits = line.strip().split("\t")
            entity = str(splits[0])
            id = int(splits[1])
            id_to_entity_embedding[id] = kelpie_entity_to_embedding[entity]

        for line in relation_file:
            splits = line.strip().split("\t")
            relation = splits[0]
            id = int(splits[1])
            id_to_relation_embedding[id] = kelpie_relation_to_embedding[relation]

        return id_to_entity_embedding, id_to_relation_embedding
