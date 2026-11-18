"""Migration wrapper tests; never start formal training."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch_geometric.data import Data
from . import handoff as h

class HandoffTests(unittest.TestCase):
    def test_archive_rejects_traversal_and_links(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/escape', tarfile.REGTYPE),
                           ('link', tarfile.SYMTYPE), ('hard', tarfile.LNKTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                archive=Path(temp)/'bad.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    item=tarfile.TarInfo(name);item.type=kind;item.linkname='/etc/passwd'
                    tar.addfile(item)
                with self.assertRaises(ValueError):h.safe_extract(archive,Path(temp)/'out')

    def test_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'good.tar.gz';target=Path(temp)/'out';target.mkdir()
            with tarfile.open(archive,'w:gz') as tar:
                item=tarfile.TarInfo('seed/last');item.size=3
                tar.addfile(item,io.BytesIO(b'abc'))
            h.safe_extract(archive,target)
            self.assertEqual((target/'seed/last').read_bytes(),b'abc')

    def test_data_relocation_preserves_registered_config(self):
        class Bundle:
            def __init__(self, config):self.config=config
        config=dict(dataset='CIFAR10',data_root='/old/cifar10',lr=.0001)
        original=dict(config)
        with patch.object(h,'OriginalBundle',Bundle),patch.object(h.p,'Bundle'):
            h.bind_data(Path('/new/datasets'));bundle=h.p.Bundle(config)
            self.assertEqual(bundle.config['data_root'],'/new/datasets/cifar10')
            self.assertEqual(bundle.config['lr'],config['lr'])
            self.assertEqual(config,original)
            self.assertIsNot(bundle.config,config)

    def test_content_fingerprint_detects_values_and_graph_order(self):
        class Dataset:
            def __init__(self):
                self._data=Data(x=torch.tensor([[1.],[2.]]),y=torch.tensor([0,1]))
                self.slices={'x':torch.tensor([0,1,2]),'y':torch.tensor([0,1,2])}
            def __len__(self):return 2
        first=Dataset();second=Dataset()
        self.assertEqual(h.fingerprint_dataset(first),h.fingerprint_dataset(second))
        second._data.x=second._data.x.flip(0)
        self.assertNotEqual(h.fingerprint_dataset(first),h.fingerprint_dataset(second))
        second=Dataset();second.slices['x'][1]=0
        self.assertNotEqual(h.fingerprint_dataset(first),h.fingerprint_dataset(second))

    def test_payload_tampering_rejected_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'last.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'payload changed'):
                h.verify_payload(root,{'files':{'last.pt':'wrong'}})

    def test_prepare_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):h.prepare(Path(temp),Path('/data'))

if __name__=='__main__':unittest.main()
