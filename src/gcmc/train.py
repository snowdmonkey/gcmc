"""Training script"""
import dgl
import torch as th
import torch.distributed
import logging
from pathlib import Path
import yaml
from .model import GCMCEncoder
from gcmc.data import MLogDataSet
from torch import Tensor
from typing import List, Tuple, Dict
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import importlib
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from .utils import prepare_mp, thread_wrapped_func

logger = logging.getLogger(__name__)


class Sampler:

    def __init__(self, g: dgl.DGLHeteroGraph, fanout: int):
        """sampler for user or item blocks

        :param g: encode graph
        :param fanout: number of neighbours for each node and each relation
        """
        self._g = g
        self._fanout = fanout

    def sample_user_blocks(self, user_nids: Tensor) -> List[dgl.DGLHeteroGraph]:
        seeds = {"user": th.unique(user_nids)}
        sub_graph = dgl.sampling.sample_neighbors(self._g, nodes=seeds, fanout=self._fanout)
        block = dgl.to_block(sub_graph, dst_nodes=seeds)
        return [block]

    def sample_item_blocks(self, item_nids: Tensor) -> List[dgl.DGLHeteroGraph]:
        seeds = {"item": th.unique(item_nids)}
        blocks = list()
        for _ in range(2):
            sub_graph = dgl.sampling.sample_neighbors(self._g, nodes=seeds, fanout=self._fanout)
            block = dgl.to_block(sub_graph, dst_nodes=seeds)
            seeds = {"user": block.srcnodes["user"].data[dgl.NID],
                     "item": block.srcnodes["item"].data[dgl.NID]}
            blocks.insert(0, block)
        return blocks


class GCMCTrainer:

    def __init__(self, model: GCMCEncoder, dataset: MLogDataSet, optimizer: Optimizer, sampler: Sampler,
                 world_size: int = 1, rank: int = 0):
        """trainer for a GCMC model

        :param model: a GCMC model to train.
        :param dataset: dataset contain encode graph, user/item features, train/valid/test labels.
        :param optimizer: optimizer for the training.
        :param sampler: a sampler to sample blocks for user or item seed nodes.
        :param world_size: number of train processes, used in multi GPU training mode.
        :param rank: index of the process for current trainer, used in multi GPU training mode.
        """
        self._model = model
        self._dataset = dataset
        self._optimizer = optimizer
        self._train_graph = dataset.train_graph
        self._sampler = sampler
        self._world_size = world_size
        self._rank = rank

    def start_train(self, n_epoch: int, batch_size: int, device: str = "cpu"):
        self._model.to(device)
        if self._world_size > 1:
            self._model = DistributedDataParallel(self._model, device_ids=[device], output_device=device)

        for epoch in range(n_epoch):
            logger.info(f"start epoch {epoch}")
            self.train_one_epoch(batch_size=batch_size, device=device)
            if self._world_size > 1:
                th.distributed.barrier()

    def train_one_epoch(self, batch_size: int, device: str):
        self._model.train()

        if self._world_size > 1:
            n_labels = self._dataset.train_labels.shape[0]
            labels = th.split(self._dataset.train_labels, n_labels // self._world_size)[self._rank]
        else:
            labels = self._dataset.train_labels
        self._train_with_labels(labels=labels, batch_size=batch_size, device=device)

    def _train_with_labels(self, labels: Tensor, batch_size: int, device: str):
        n_label = labels.size()[0]
        for i, batch_start_index in enumerate(range(0, n_label, batch_size)):
            positive_label = self._dataset.train_labels[batch_start_index:(batch_start_index + batch_size)]
            labels = self._prepare_labels(positive_label)
            user_nf, item_nf = self._prepare_blocks(labels, device)
            loss = self._train_one_batch(labels, user_nf, item_nf, device=device)
            logger.info(f"batch {i}; loss={loss}")

    def _sample_negative_label(self, positive_label: Tensor) -> Tensor:
        negative_label = positive_label.clone()
        negative_label[:, 1] = th.randint(
            low=0,
            high=self._dataset.n_items,
            size=(negative_label.size()[0],)
        )
        negative_label[:, 2] = 0
        return negative_label

    def _copy_feature_to_blocks(self, block: dgl.DGLHeteroGraph, device: str):
        for node_type in ["user", "item"]:
            feature_name = f"{node_type}_features"
            if block.number_of_src_nodes(node_type) > 0:
                if node_type == "user":
                    features = self._dataset.user_features[block.srcnodes[node_type].data[dgl.NID]]
                else:
                    features = self._dataset.item_features[block.srcnodes[node_type].data[dgl.NID]]
                block.srcnodes[node_type].data[feature_name] = features.to(device)
                block.srcnodes[node_type].data[dgl.NID] = block.srcnodes[node_type].data[dgl.NID].to(device)
            if block.number_of_dst_nodes(node_type) > 0:
                if node_type == "user":
                    features = self._dataset.user_features[block.dstnodes[node_type].data[dgl.NID]]
                else:
                    features = self._dataset.item_features[block.dstnodes[node_type].data[dgl.NID]]
                block.dstnodes[node_type].data[feature_name] = features.to(device)
                block.dstnodes[node_type].data[dgl.NID] = block.dstnodes[node_type].data[dgl.NID].to(device)

    def _prepare_labels(self, positive_label: Tensor):
        positive_label = th.cat([
            positive_label, th.ones(positive_label.shape[0], 1, dtype=positive_label.dtype)
        ], dim=1)
        negative_label = self._sample_negative_label(positive_label)
        return th.cat([positive_label, negative_label], dim=0)

    def _prepare_blocks(self, labels: Tensor, device: str) -> Tuple[List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph]]:
        user_seeds = labels[:, 0]
        item_seeds = labels[:, 1]
        user_blocks = self._sampler.sample_user_blocks(user_seeds)
        item_blocks = self._sampler.sample_item_blocks(item_seeds)
        for block in user_blocks:
            self._copy_feature_to_blocks(block, device)
        for block in item_blocks:
            self._copy_feature_to_blocks(block, device)
        return user_blocks, item_blocks

    def _train_one_batch(self, labels: Tensor, user_blocks: List[dgl.DGLHeteroGraph],
                         item_blocks: List[dgl.DGLHeteroGraph], device: str):
        labels = labels.to(device)
        user_embedding, user_nid_mapping = self._model(user_blocks, "user")
        item_embedding, item_nid_mapping = self._model(item_blocks, "item")
        user_index = [user_nid_mapping[user_nid] for user_nid in labels[:, 0].tolist()]
        item_index = [item_nid_mapping[item_nid] for item_nid in labels[:, 1].tolist()]
        user_embeds = user_embedding[user_index]
        item_embeds = item_embedding[item_index]
        logits = (user_embeds * item_embeds).sum(1)
        loss = F.binary_cross_entropy_with_logits(logits, labels[:, 2].double())
        self._optimizer.zero_grad()
        loss.backward()
        if self._world_size > 1:
            for param in self._model.parameters():
                if param.requires_grad and param.grad is not None:
                    th.distributed.all_reduce(param.grad.data, op=th.distributed.ReduceOp.SUM)
        self._optimizer.step()
        return loss.item()


def train_in_mp(process_index: int, n_processes: int, dataset: MLogDataSet, config: Dict):
    device = config["train"]["device"][process_index]

    logger.info(f"start process with device {device}")
    rank = process_index
    dist_init_method = "tcp://127.0.0.1:12345"
    th.distributed.init_process_group(backend="nccl", init_method=dist_init_method, world_size=n_processes,
                                      rank=rank)
    th.cuda.synchronize()

    model = GCMCEncoder(n_users=dataset.n_users, n_items=dataset.n_items, **config["model"])

    optimizer = getattr(importlib.import_module("torch.optim"), config["optimizer"].pop("class"))(
        params=model.parameters(), **config["optimizer"]
    )
    sampler = Sampler(g=dataset.train_graph, **config["sampler"])
    gcmc_trainer = GCMCTrainer(model=model, dataset=dataset, optimizer=optimizer,
                               sampler=sampler, world_size=n_processes, rank=rank)
    gcmc_trainer.start_train(n_epoch=config["train"]["n_epoch"], batch_size=config["train"]["batch_size"],
                             device=device)


def train_with_multi_gpu(config: Dict):
    logger.info("start to load dataset")
    dataset = MLogDataSet.load(Path(config["data"]["mlog_data_path"]))
    devices: List = config["train"]["device"]

    logger.info("start to prepare graph for share memory")
    prepare_mp(dataset.train_graph)

    processes = list()
    for process_index in range(len(devices)):
        p = mp.Process(target=thread_wrapped_func(train_in_mp), args=(process_index, len(devices), dataset, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def train_with_single_device(config: Dict):
    logger.info(f"start to load dataset")
    dataset = MLogDataSet.load(Path(config["data"]["mlog_data_path"]))
    logger.info(f"number of users = {dataset.n_users:,}, number of items = {dataset.n_items:,}")

    model = GCMCEncoder(n_users=dataset.n_users, n_items=dataset.n_items, **config["model"])

    optimizer = getattr(importlib.import_module("torch.optim"), config["optimizer"].pop("class"))(
        params=model.parameters(), **config["optimizer"]
    )
    sampler = Sampler(g=dataset.train_graph, **config["sampler"])
    gcmc_trainer = GCMCTrainer(model=model, dataset=dataset, optimizer=optimizer,
                               sampler=sampler)
    gcmc_trainer.start_train(**config["train"])


def main():
    import argparse
    parser = argparse.ArgumentParser("train GCMC model with a yaml config file")
    parser.add_argument("-c", dest="config_path", type=lambda x: Path(x))

    args = parser.parse_args()
    with args.config_path.open("r") as f:
        config = yaml.safe_load(f)

    if isinstance(config["train"]["device"], list):
        train_with_multi_gpu(config)
    else:
        train_with_single_device(config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
    main()
