import torch
from .Model import Model
torch.cuda.empty_cache()
class ConvE(Model):
    def __init__(self, ent_total, rel_total, dims, hidden_layer_size=9728):
        super(ConvE, self).__init__(ent_tot=ent_total, rel_tot=rel_total)

        # self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.dim = dims
        self.embedding_width = 20
        self.embedding_height = self.dim // self.embedding_width
        self.hidden_layer_size = hidden_layer_size

        self.conv_kernel_shape = (3, 3)  # convolutional kernel shape
        self.num_conv_filters = 32  # number of convolutional filters

        self.batch_norm_1 = torch.nn.BatchNorm2d(1).cuda()
        self.batch_norm_2 = torch.nn.BatchNorm2d(self.num_conv_filters).cuda()
        self.batch_norm_3 = torch.nn.BatchNorm1d(self.dim).cuda()
        self.convolutional_layer = torch.nn.Conv2d(1, self.num_conv_filters, self.conv_kernel_shape, 1, 0,
                                                   bias=True).cuda()
        self.hidden_layer = torch.nn.Linear(self.hidden_layer_size, self.dim).cuda()

    def initialize_model(self):
        self.create_embedding(dimension=self.dim, emb_type="entity", name="e")
        # self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.create_embedding(dimension=self.dim, emb_type="relation", name="r")
    def _calc(self, head_emb, rel_emb, tail_emb):
        e1_embedded = head_emb.view(-1, 1, self.embedding_width, self.embedding_height)
        rel_embedded = rel_emb.view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.inp_drop(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_drop(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_drop(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)
        scores = torch.mm(x, tail_emb.transpose(1, 0))

        # x += self.b.expand_as(x)
        scores = torch.sigmoid(scores)
        output_scores = torch.diagonal(scores)

        return output_scores

    def return_score(self, is_predict=True):
        (head_emb, rel_emb, tail_emb) = self.current_batch
        head = head_emb["e"]
        rel = rel_emb["r"]
        tail = tail_emb["e"]
        return self._calc(head, rel, tail)
