import unittest
from .selection import parse_table, select


class SelectionTests(unittest.TestCase):
    def test_metric_direction_and_rounded_small_gap(self):
        rows = dict(QAT_CIFAR10_GCN=dict(gpf_lorap=dict(mean=43.6), graphlora=dict(mean=43.62)),
                    DQ_ZINC_GCN=dict(gpf_lorap=dict(mean=.570), graphlora=dict(mean=.5676)),
                    A2Q_Cora_GIN=dict(gpf_lorap=dict(mean=78.5), graphlora=dict(mean=66.8)),
                    A2Q_ZINC_GIN=dict(gpf_lorap=dict(mean=.361), graphlora=dict(mean=.4617)),
                    QAT_MNIST_GAT=dict(gpf_lorap=dict(mean=30.), graphlora=dict(mean=30.)))
        self.assertEqual(select(rows), ['QAT_CIFAR10_GCN', 'DQ_ZINC_GCN'])

    def test_incomplete_table_rejected(self):
        with self.assertRaises(ValueError):
            parse_table('% QAT + GIN\n & & GPF-LoRAP & $1\\pm2$ \\\\')


if __name__ == '__main__':
    unittest.main()
