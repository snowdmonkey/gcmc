import pickle
import random
from typing import List, Tuple, Dict, Union, Set, Optional
from pathlib import Path
import logging

import dgl
import torch
from torch import Tensor
from multiprocessing import Pool, Manager
from dgl import DGLHeteroGraph

logger = logging.getLogger(__name__)

POSSIBLE_RATINGS = [0, 1, 2, 3, 4, 5, 6]

UserItemRating = Tuple[int, int, Set[int]]


class MLogDataSet:
    """A class to store the raw information for mlog recommendation, including user features, item features, user to
    item relations, user id to node id mapping, item id to node id mapping
    """

    def __init__(self, train_graph: DGLHeteroGraph, user_id_to_nid: Dict[int, int],
                 item_id_to_nid: Dict[int, int], user_features: Tensor, item_features: Tensor, train_labels: Tensor,
                 valid_labels: Tensor, test_labels: Tensor):
        """
        :param train_graph: a dgl hetero graph, contain all the users, items and links in train labels.
        :param user_id_to_nid: a dict mapping user_id to user node id
        :param item_id_to_nid: a dict mapping item_id to item node id
        :param user_features: user features, tensor of dim n_user*user_feature_dim
        :param item_features: item features, tensor of dim n_item*item_feature_dim
        """
        self._train_graph = train_graph
        self._user_id_to_nid = user_id_to_nid
        self._item_id_to_nid = item_id_to_nid
        self._n_users = len(user_id_to_nid)
        self._n_items = len(item_id_to_nid)

        self._user_features = user_features
        self._item_features = item_features

        self._train_labels = train_labels
        self._valid_labels = valid_labels
        self._test_labels = test_labels

    @classmethod
    def from_dat_files(cls, rating_path: Path, user_feature_path: Path, item_feature_path: Path, valid_ratio: float,
                       test_ratio: float):
        """create mlog dataset from raw dat files

        :param test_ratio: test labels ratio in term of lines in raw rating file
        :param valid_ratio: valid labels ratio in term of lines in raw rating file
        :param rating_path: dat file contain raw ratings, line format: "{user_id}\t{item_id}\t{rating1},{rating2},..."
        :param user_feature_path: dat file contain raw user features, line format "{user_id}\t{feature1},{feature2},..."
        :param item_feature_path: dat file contain raws item features, line format "{item_id}\t{item1},{item2},..."
        :return: instance of MLogDataset
        """
        logger.info("start to process user feature file")
        user_id_to_nid, user_feature = cls._process_feature_file(user_feature_path)
        logger.info("start to process item feature file")
        item_id_to_nid, item_feature = cls._process_feature_file(item_feature_path)
        logger.info("start to process rating file")
        train_graph, train_label, valid_label, test_label = \
            cls._process_rating_file(rating_path, user_id_to_nid, item_id_to_nid, valid_ratio, test_ratio)
        logger.info("Complete processing rating file")
        return cls(train_graph, user_id_to_nid, item_id_to_nid, user_feature, item_feature, train_label, valid_label,
                   test_label)

    @classmethod
    def _process_rating_file(cls, rating_path: Path, user_mapping: Dict[int, int], item_mapping: Dict[int, int],
                             valid_ratio: float, test_ratio: float) \
            -> Tuple[DGLHeteroGraph, Tensor, Tensor, Tensor]:
        chunk_size = 20_000_000
        train_labels: Optional[Tensor] = None
        valid_labels: Optional[Tensor] = None
        test_labels: Optional[Tensor] = None
        relations: Dict[int, Tensor] = dict()
        with rating_path.open("r") as f:
            lines = f.readlines()
        n_lines = len(lines)
        n_chunks = n_lines // chunk_size + 1
        for i_chunk in range(n_chunks):
            chunk_start = i_chunk*chunk_size
            chunk_end = min(chunk_start+chunk_size, n_lines)
            logger.info(f"start to process line {chunk_start:,} to line {chunk_end:,}")
            if chunk_start >= n_lines:
                break
            else:
                chunk_train_label, chunk_valid_label, chunk_test_label, chunk_relation = \
                    cls._process_chunk_of_lines(
                        lines[chunk_start:chunk_end], user_mapping, item_mapping, valid_ratio, test_ratio)
            if train_labels is None:
                train_labels = chunk_train_label
            else:
                train_labels = torch.cat([train_labels, chunk_train_label])
            if valid_labels is None:
                valid_labels = chunk_valid_label
            else:
                valid_labels = torch.cat([valid_labels, chunk_valid_label])
            if test_labels is None:
                test_labels = chunk_test_label
            else:
                test_labels = torch.cat([test_labels, chunk_test_label])
            for k, v in chunk_relation.items():
                if relations.get(k) is None:
                    relations.update({k: v})
                else:
                    relations[k] = torch.cat([relations[k], v])
        return cls._relation_to_graph(relations, len(user_mapping), len(item_mapping)), \
               train_labels, valid_labels, test_labels

    @classmethod
    def _relation_to_graph(cls, relations: Dict[int, Tensor], n_users: int, n_items: int):
        bipartite_graphs = list()
        for relation, user_item_pairs in relations.items():
            if relation not in POSSIBLE_RATINGS:
                continue
            user_nids = user_item_pairs[:, 0].tolist()
            item_nids = user_item_pairs[:, 1].tolist()
            bipartite_graphs.append(dgl.bipartite(
                (user_nids, item_nids), "user", str(relation), "item", card=(n_users, n_items)))
            bipartite_graphs.append(dgl.bipartite(
                (item_nids, user_nids), "item", f"rev-{relation}", "user", card=(n_items, n_users)))
        return dgl.hetero_from_relations(bipartite_graphs)

    @classmethod
    def _process_chunk_of_lines(cls, lines: List[str], user_mapping: Dict[int, int], item_mapping: Dict[int, int],
                                valid_ratio: float, test_ratio: float) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        with Pool() as p:
            user_item_ratings: List[UserItemRating] = p.map(cls._process_rating_file_line, lines)
        n_lines = len(lines)
        n_valid = int(n_lines*valid_ratio)
        n_test = int(n_lines*test_ratio)
        n_train = n_lines - n_valid - n_test
        group_indicator = [0]*n_train+[1]*n_valid+[2]*n_test
        random.shuffle(group_indicator)
        train_label: List[Tuple[int, int]] = list()
        valid_label: List[Tuple[int, int]] = list()
        test_label: List[Tuple[int, int]] = list()
        relations: Dict[int, Union[List[Tuple[int, int]], Tensor]] = dict()
        for group, (user_id, item_id, ratings) in zip(group_indicator, user_item_ratings):
            try:
                user_nid = user_mapping[user_id]
                item_nid = item_mapping[item_id]
            except KeyError:
                # logger.error(f"no such user/item id feature {user_id} {item_id}")
                continue
            if group == 1:
                valid_label.append((user_nid, item_nid))
            elif group == 2:
                test_label.append((user_nid, item_nid))
            else:
                assert group == 0
                train_label.append((user_nid, item_nid))
                for rating in ratings:
                    if relations.get(rating) is None:
                        relations.update({rating: list()})
                    relations[rating].append((user_nid, item_nid))
        for rating in relations:
            relations[rating] = torch.tensor(relations[rating], dtype=torch.int64)
        return torch.tensor(train_label, dtype=torch.int64), \
               torch.tensor(valid_label, dtype=torch.int64), \
               torch.tensor(test_label, dtype=torch.int64), \
               relations

    @classmethod
    def _process_rating_file_line(cls, line: str) -> UserItemRating:
        words = line.split("\t")
        user_id, item_id = int(words[0]), int(words[1])
        ratings = {int(x) for x in words[2].split(",")}
        return user_id, item_id, ratings

    @classmethod
    def _process_feature_file(cls, feature_file_path: Path) -> Tuple[dict, Tensor]:
        """read feature file, create user/item id to user/item nid mapping and feature tensor"""
        with feature_file_path.open("r") as f:
            lines = f.readlines()
        with Pool() as pool:
            lines = pool.map(cls._process_feature_dat_line, lines)
        id_to_nid = dict()
        features = list()
        for i, (id_, feature) in enumerate(lines):
            id_to_nid.update({id_: i})
            features.append(feature)
        return id_to_nid, torch.tensor(features)

    @classmethod
    def _process_feature_dat_line(cls, line: str) -> Tuple[int, List[float]]:
        words = line.split("\t")
        id_ = int(words[0])
        feature_value = [float(x) for x in words[1].split(",")]
        return id_, feature_value

    @property
    def user_id_to_nid(self):
        return self._user_id_to_nid

    @property
    def item_id_to_nid(self):
        return self._item_id_to_nid

    @property
    def n_users(self) -> int:
        return self._n_users

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def train_graph(self) -> DGLHeteroGraph:
        return self._train_graph

    @property
    def user_features(self) -> Tensor:
        return self._user_features

    @property
    def item_features(self) -> Tensor:
        return self._item_features

    @property
    def train_labels(self) -> Tensor:
        return self._train_labels

    @property
    def valid_labels(self) -> Tensor:
        return self._valid_labels

    @property
    def test_labels(self) -> Tensor:
        return self._test_labels

    def dump(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        graph_path = path / "graph"
        tensor_path = path / "tensors"
        with graph_path.open("wb") as f:
            pickle.dump(self._train_graph, f, protocol=4)
        torch.save({
            "user_id_to_nid": self._user_id_to_nid,
            "item_id_to_nid": self._item_id_to_nid,
            "user_features": self._user_features,
            "item_features": self._item_features,
            "train_labels": self._train_labels,
            "valid_labels": self._valid_labels,
            "test_labels": self._test_labels
        }, str(tensor_path))

    @classmethod
    def load(cls, path: Path):
        graph_path = path / "graph"
        tensor_path = path / "tensors"
        with graph_path.open("rb") as f:
            train_graph = pickle.load(f)
        return cls(train_graph = train_graph, **torch.load(tensor_path))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    import argparse
    parser = argparse.ArgumentParser("User item rating data process")
    subparsers = parser.add_subparsers(dest="subparser_name", help="mlog for generate mlog dataset, "
                                            "split for generate a train, validate, test splits.")
    parser_mlog = subparsers.add_parser("mlog", help="create mlog dataset")
    parser_mlog.add_argument("--rating", type=lambda x: Path(x), help="Path to rating file")
    parser_mlog.add_argument("--user-feature", type=lambda x: Path(x), help="Path to user feature file")
    parser_mlog.add_argument("--item-feature", type=lambda x: Path(x), help="Path to item feature file")
    parser_mlog.add_argument("--valid_ratio", type=float)
    parser_mlog.add_argument("--test_ratio", type=float)
    parser_mlog.add_argument("--dump", type=lambda x: Path(x), help="Path to dump the mlog dataset")

    args = parser.parse_args()

    if args.subparser_name == "mlog":
        dataset = MLogDataSet.from_dat_files(
            args.rating, args.user_feature, args.item_feature, args.valid_ratio, args.test_ratio)
        dataset.dump(args.dump)

