"""NN modules"""
import logging
from typing import List, Dict, Tuple
import dgl
import dgl.function as dgl_fn
from dgl import DGLHeteroGraph
import torch as th
import torch.nn as nn

logger = logging.getLogger()

node_embed_names = ['agg_ebd']


class GCMCEncoder(nn.Module):
    def __init__(self, rating_set: List[int], n_items: int, n_users: int, embedding_dim: int, user_feature_dim: int,
                 item_feature_dim: int, dropout_rate: float = 0.0):
        """GCMC encoder network used to calculate user and item embeddings

        :param rating_set: list of rating values we are going to leverage for information propagation
        :param n_items: number of item nodes in graph
        :param n_users: number of user nodes in graph
        :param embedding_dim: dimension of user/item embeddings
        :param user_feature_dim: dimension of user feature
        :param item_feature_dim: dimension of item feature
        :param dropout_rate: drop our rate
        """
        super().__init__()
        self._n_items = n_items
        self._n_users = n_users
        self._embedding_dim = embedding_dim
        self._user_feature_dim = user_feature_dim
        self._item_feature_dim = item_feature_dim
        self._rating_set = [str(rating) for rating in rating_set]
        self._dropout = nn.Dropout(dropout_rate)
        self._item_id_embedding_layer = nn.Embedding(self._n_items, self._embedding_dim)
        self._n_rating_set = len(rating_set)

        self._W = nn.ModuleDict()
        for rating in self._rating_set:
            # self._etypes_need_compute.extend([rating, f"rev-{rating}"])
            self._W.update({
                rating: nn.Linear(self._embedding_dim, self._embedding_dim),
                f"rev-{rating}": nn.Linear(self._embedding_dim + self._item_feature_dim, self._embedding_dim)
            })

        self._item_aggregate_layer = nn.Linear(embedding_dim * self._n_rating_set+self._item_feature_dim, embedding_dim)
        self._user_aggregate_layer = nn.Linear(embedding_dim * self._n_rating_set+self._user_feature_dim, embedding_dim)

        # initialize parameters
        self.reset_parameters()

        self._agg_activation = nn.LeakyReLU()

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _propagate_user_blocks(self, blocks: List[DGLHeteroGraph]) -> th.Tensor:
        assert len(blocks) == 1
        user_embeddings = self._propagate_item_to_user(blocks[0])
        return user_embeddings

    def _propagate_item_blocks(self, blocks: List[DGLHeteroGraph]) -> th.Tensor:
        assert len(blocks) == 2
        user_embeddings = self._propagate_item_to_user(blocks[0])
        blocks[1].srcnodes["user"].data["h"] = user_embeddings
        item_embeddings = self._propagate_user_to_item(blocks[1])
        return item_embeddings

    def _propagate_item_to_user(self, block: DGLHeteroGraph) -> th.Tensor:
        with block.local_scope():
            block.srcnodes["item"].data["item_id_embedding"] = \
                self._item_id_embedding_layer(block.srcnodes["item"].data[dgl.NID])
            for etype in [f"rev-{rating}" for rating in self._rating_set]:
                block.apply_edges(lambda edges: self._compute_message_item_to_user(edges, etype), etype=etype)
                block.update_all(dgl_fn.copy_e("m", f"m_{etype}"), dgl_fn.mean(f"m_{etype}", f"h_{etype}"), etype=etype)
            user_feature = block.dstnodes["user"].data["user_features"]
            all_features_on_user = [user_feature]
            for rating in self._rating_set:
                feature_name = f"h_rev-{rating}"
                if feature_name in block.dstnodes["user"].data:
                    all_features_on_user.append(block.dstnodes["user"].data[feature_name])
                else:
                    all_features_on_user.append(th.zeros(
                        user_feature.shape[0], self._embedding_dim,
                        dtype=user_feature.dtype, device=user_feature.device))

            return self._agg_activation(self._user_aggregate_layer(th.cat(all_features_on_user, dim=1)))

    def _propagate_user_to_item(self, block: DGLHeteroGraph) -> th.Tensor:
        with block.local_scope():
            for etype in self._rating_set:
                block.apply_edges(lambda edges: self._compute_message_user_to_item(edges, etype), etype=etype)
                block.update_all(dgl_fn.copy_e("m", f"m_{etype}"), dgl_fn.mean(f"m_{etype}", f"h_{etype}"), etype=etype)
            item_features: th.Tensor = block.dstnodes["item"].data["item_features"]
            all_feature_on_item = [item_features]
            for rating in self._rating_set:
                feature_name = f"h_{rating}"
                if feature_name in block.dstnodes["item"].data:
                    all_feature_on_item.append(block.dstnodes["item"].data[feature_name])
                else:
                    all_feature_on_item.append(th.zeros(
                        item_features.shape[0], self._embedding_dim,
                        dtype=item_features.dtype, device=item_features.device))

            return self._agg_activation(self._item_aggregate_layer(th.cat(all_feature_on_item, dim=1)))

    def _compute_message_item_to_user(self, edges: dgl.EdgeBatch, etype: str):
        return {
            "m": self._W[etype](th.cat([edges.src["item_features"], edges.src["item_id_embedding"]], dim=1))
        }

    def _compute_message_user_to_item(self, edges: dgl.EdgeBatch, etype: str):
        return {
            "m": self._W[etype](edges.src["h"])
        }

    def forward(self, blocks: List[DGLHeteroGraph], dst_node_type: str) -> Tuple[th.Tensor, Dict[int, int]]:
        """forward function

        :param blocks: list of dgl.to_block results, if dst_node_type is "user", blocks length should be 1, if
            dst_node_type is "item", block length should be 2
        :param dst_node_type: either "user" or "item"
        :return: user/item embeddings and a dict to map user/item nid to the row index in embedding tensor corresponding
            to it.
        """
        if dst_node_type == "user":
            user_embedding = self._propagate_user_blocks(blocks)
            user_nids = blocks[-1].dstnodes["user"].data[dgl.NID].tolist()
            nid_to_embedding_index = {nid: index for index, nid in enumerate(user_nids)}
            return user_embedding, nid_to_embedding_index
        elif dst_node_type == "item":
            item_embedding = self._propagate_item_blocks(blocks)
            item_nids = blocks[-1].dstnodes["item"].data[dgl.NID].tolist()
            nid_to_embedding_index = {nid: index for index, nid in enumerate(item_nids)}
            return item_embedding, nid_to_embedding_index
        else:
            raise ValueError()
