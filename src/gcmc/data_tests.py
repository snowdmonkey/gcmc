import unittest
from gcmc.data import MLogDataSet
from pathlib import Path
from dgl import DGLHeteroGraph
import dgl


class MLogDataSetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.rating_path = Path("data/test/rating.dat")
        self.user_feature_path = Path("data/test/user_feature.dat")
        self.item_feature_path = Path("data/test/item_feature.dat")

    def test_create_from_dat(self):
        dataset = MLogDataSet.from_dat_files(self.rating_path, self.user_feature_path, self.item_feature_path, 0.1, 0.1)
        self.assertEqual(dataset._train_labels.shape, (8000, 2))

    def test_dump_mlogdataset(self):
        dataset = MLogDataSet.from_dat_files(self.rating_path, self.user_feature_path, self.item_feature_path, 0.1, 0.1)
        dataset.dump(Path("data/test/test_dump"))
        dataset_loaded = MLogDataSet.load(Path("data/test/test_dump"))
        self.assertEqual(dataset_loaded._valid_labels.shape, (1000, 2))

    def test_generate_train_split(self):
        dataset = MLogDataSet.load(Path("data/test/test_dump"))
        g = dataset._train_graph
        user_to_item_etyps = list()
        item_to_user_etypes = list()
        for u, e, v in g.canonical_etypes:
            if u == "user":
                user_to_item_etyps.append(e)
            else:
                item_to_user_etypes.append(e)

        for label in dataset._train_labels:
            label = label.tolist()
            if any(g.has_edge_between(label[0], label[1], etype=x) for x in user_to_item_etyps) is not True:
                raise Exception()
            self.assertTrue(any(g.has_edge_between(label[0], label[1], etype=x) for x in user_to_item_etyps))
            self.assertTrue(any(g.has_edge_between(label[1], label[0], etype=x) for x in item_to_user_etypes))
        for label in dataset._valid_labels:
            label = label.tolist()
            self.assertFalse(any(g.has_edge_between(label[0], label[1], etype=x) for x in user_to_item_etyps))
            self.assertFalse(any(g.has_edge_between(label[1], label[0], etype=x) for x in item_to_user_etypes))
        for label in dataset._test_labels:
            label = label.tolist()
            self.assertFalse(any(g.has_edge_between(label[0], label[1], etype=x) for x in user_to_item_etyps))
            self.assertFalse(any(g.has_edge_between(label[1], label[0], etype=x) for x in item_to_user_etypes))


if __name__ == '__main__':
    unittest.main()
