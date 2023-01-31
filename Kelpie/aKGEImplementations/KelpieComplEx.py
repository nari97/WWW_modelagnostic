from AugmentedKGE.Models.ComplEx import ComplEx
import torch


class KelpieComplEx(ComplEx):

    def __init__(self, ent_total, rel_total, dim, norm=2):
        super(KelpieComplEx, self).__init__(ent_total, rel_total, dim)

    def load_kelpie_model(self, id_to_entity, id_to_relation):
        """
            Loads embeddings from Kelpie implementation of TransE to aKGE implementation of TransE

            Parameters:
                id_to_entity (Dict): Mapping from aKGE dataset entity id to kelpie embedding
                id_to_relation (Dict): Mapping from aKGE dataset relation id to kelpie embedding
        """
        entity_embedding_real = torch.rand(self.ent_tot, self.dim)
        entity_embedding_img = torch.rand(self.ent_tot, self.dim)
        relation_embedding_real = torch.rand(self.rel_tot, self.dim)
        relation_embedding_img = torch.rand(self.rel_tot, self.dim)

        for i in range(0, self.ent_tot):
            entity_embedding_real[i, :] = id_to_entity[i][0:self.dim]
            entity_embedding_img[i, :] = id_to_entity[i][self.dim:]

        for i in range(0, self.rel_tot):
            relation_embedding_real[i, :] = id_to_relation[i][0:self.dim]
            relation_embedding_img[i, :] = id_to_relation[i][self.dim:]

        self.embeddings["entity"]["e_real"].emb.data = entity_embedding_real
        self.embeddings["entity"]["e_img"].emb.data = entity_embedding_img
        self.embeddings["relation"]["r_real"].emb.data = relation_embedding_real
        self.embeddings["relation"]["r_img"].emb.data = relation_embedding_img

    def test_kelpie_load(self, id_to_entity, id_to_relation):

        flag_entity = True
        flag_relation = True

        for i in range(self.ent_tot):
            if not torch.equal(id_to_entity[i][0:self.dim], self.embeddings["entity"]["e_real"].emb.data[i, :]) and not torch.equal(id_to_entity[i][self.dim:], self.embeddings["entity"]["e_img"].emb.data[i, :]):
                flag_entity = False
                break

        for i in range(self.rel_tot):
            if not torch.equal(id_to_relation[i][0:self.dim], self.embeddings["relation"]["r_real"].emb.data[i, :]) and not torch.equal(id_to_relation[i][self.dim:], self.embeddings["relation"]["r_img"].emb.data[i, :]):
                flag_relation = False
                break

        print("Entity load successful:", flag_entity)
        print("Relation load successful:", flag_relation)