from AugmentedKGE.Models.ConvE import ConvE
import torch

class KelpieConvE(ConvE):

    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(KelpieConvE, self).__init__(ent_total, rel_total, dim)

    def load_kelpie_model(self, id_to_entity, id_to_relation):
        """
            Loads embeddings from Kelpie implementation of TransE to aKGE implementation of TransE

            Parameters:
                id_to_entity (Dict): Mapping from aKGE dataset entity id to kelpie embedding
                id_to_relation (Dict): Mapping from aKGE dataset relation id to kelpie embedding
        """

        entity_embedding = torch.rand(self.ent_tot, self.dim)
        relation_embedding = torch.rand(self.rel_tot, self.dim)

        for i in range(0, self.ent_tot):
            entity_embedding[i, :] = id_to_entity[i]

        for i in range(0, self.rel_tot):
            relation_embedding[i, :] = id_to_relation[i]

        self.embeddings["entity"]["e"].emb.data = entity_embedding
        self.embeddings["relation"]["r"].emb.data = relation_embedding

    def test_kelpie_load(self, id_to_entity, id_to_relation):

        flag_entity = True
        flag_relation = True

        for i in range(self.ent_tot):
            if not torch.equal(id_to_entity[i], self.embeddings["entity"]["e"].emb.data[i, :]):
                flag_entity = False
                break

        for i in range(self.rel_tot):
            if not torch.equal(id_to_relation[i], self.embeddings["relation"]["r"].emb.data[i, :]):
                flag_relation = False
                break

        print("Entity load successful:", flag_entity)
        print("Relation load successful:", flag_relation)