import torch
from .Model import Model


class TuckER(Model):

    def __init__(self, ent_total, rel_total, dim_e, dim_r, input_d, hidden_d1, hidden_d2):
        """
        Args:
            ent_total (int): Total number of entities
            rel_total (int): Total number of relations
            dim_e (int): Number of dimensions for entity embeddings
            dim_r (int): Number of dimensions for relation embeddings
        """
        super(TuckER, self).__init__(ent_total, rel_total)
        self.dim_e = dim_e
        self.dim_r = dim_r

        self.input_dropout = torch.nn.Dropout(input_d)
        self.hidden_dropout1 = torch.nn.Dropout(hidden_d1)
        self.hidden_dropout2 = torch.nn.Dropout(hidden_d2)

        self.bn0 = torch.nn.BatchNorm1d(dim_e)
        self.bn1 = torch.nn.BatchNorm1d(dim_e)

    def get_default_loss(self):
        return 'bce'

    def initialize_model(self):
        self.create_embedding(self.dim_e, emb_type="entity", name="e")
        self.create_embedding(self.dim_r, emb_type="relation", name="r")
        self.create_embedding((self.dim_r, self.dim_e, self.dim_e), emb_type="global", name="w", init="kaiming_uniform")

    def _calc(self, h, r, t, w, is_predict):
        e1 = h
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W = w
        W_mat = torch.mm(r, W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.embeddings["entity"]["e"].emb.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def return_score(self, is_predict=False):
        (head_emb, rel_emb, tail_emb) = self.current_batch

        h = head_emb["e"]
        t = tail_emb["e"]
        r = rel_emb["r"]
        w = self.current_global_embeddings["w"]

        return self._calc(h, r, t, w, is_predict)
