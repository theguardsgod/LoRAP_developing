"""Protocol identity, paired differences and the supplemental test-release barrier."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .run import write_json
from .table_v_run import table_config
from .table_v_control import compare_pair, inherited_config, worker


class ControlTests(unittest.TestCase):
    def test_only_variant_changes_in_formal_protocol(self):
        for mode in ('fixed4','a2q'):
            source=table_config(mode,list(range(41,51)),200)
            control=inherited_config(source)
            self.assertEqual({k for k in source if source[k]!=control[k]}, {'variants'})
            self.assertEqual(control['variants'],['no_prompt'])
            self.assertEqual(source['variants'],['lora'])
            smoke=inherited_config(source,smoke=True)
            self.assertEqual(smoke['epochs'],2)
            self.assertEqual(smoke['seeds'],[41])
            self.assertEqual(smoke['subset_sizes'],dict(train=128,val=64,test=64))

    def test_pair_direction_and_pairing_validation(self):
        common=dict(dataset='MNIST',seed=41,epochs=200,common_backbone_sha256='same',test_evaluations=1)
        control=dict(common,prompt_parameters=0,lora_parameters=0,total_parameters=10000,trainable_parameters=9000,
                     test=dict(accuracy=.95,loss=.2))
        lora=dict(common,lora_parameters=4426,total_parameters=14426,trainable_parameters=13426,
                  test=dict(accuracy=.94,loss=.21))
        pair=compare_pair(lora,control)
        self.assertAlmostEqual(pair['lora_minus_no_prompt_pp'],-1)
        self.assertAlmostEqual(pair['lora_minus_no_prompt_CE'],.01)
        for bad in (dict(control,common_backbone_sha256='different'),dict(control,seed=42),
                    dict(control,epochs=199),dict(control,prompt_parameters=1)):
            with self.assertRaises(AssertionError):
                compare_pair(lora,bad)

    def test_no_control_test_until_both_protocols_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_json(root/'suite.json',dict(seeds=[41],threads=2,device='cuda:0',smoke=False))
            for mode in ('fixed4','a2q'):
                config=inherited_config(table_config(mode,[41],200))
                write_json(root/mode/'MNIST/config.json',config)
                write_json(root/'reference'/mode/'MNIST/split_manifest.json',{})
                write_json(root/'reference'/mode/'MNIST/lora/seed_41/result.json',dict(common_backbone_sha256='same'))
            events=[]
            def fitted(path,*args):
                events.append('fit-'+path.name)
                return dict(common_backbone_sha256='same')
            with patch('runs.lorap_study.table_v_control.require_device'), \
                 patch('runs.lorap_study.table_v_control.study_data',return_value=({},{})), \
                 patch('runs.lorap_study.table_v_control.fit',side_effect=fitted), \
                 patch('runs.lorap_study.table_v_control.release_tests',side_effect=lambda p,c: events.append('release-'+p.name)), \
                 patch('runs.lorap_study.table_v_control.test_once',side_effect=lambda p,*a: events.append('test-'+p.name)):
                worker(root)
            self.assertEqual(events,['fit-fixed4','fit-a2q','release-fixed4','release-a2q','test-fixed4','test-a2q'])
            with patch('runs.lorap_study.table_v_control.require_device'), \
                 patch('runs.lorap_study.table_v_control.study_data',return_value=({},{})), \
                 patch('runs.lorap_study.table_v_control.fit',side_effect=RuntimeError('failed')), \
                 patch('runs.lorap_study.table_v_control.release_tests') as release, \
                 patch('runs.lorap_study.table_v_control.test_once') as evaluate:
                with self.assertRaises(RuntimeError):
                    worker(root)
                release.assert_not_called()
                evaluate.assert_not_called()


if __name__=='__main__':
    unittest.main()
