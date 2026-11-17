"""Release contracts use synthetic fixtures, not the paper's sealed test."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .legacy_world import SFWorldExact
from .paper_aligned_lower import ContextActionSF, candidate_values, sf_batch_loss
from .paper_aligned_run import _entry_to_primitive
from .paper_aligned_attention import PaperAlignedAttention
from .public_run import freeze, evaluate_test, new_json, protocol_identity, sha
from .task_protocol import create_protocol, load_tasks

class ConstantSF(torch.nn.Module):
    def __init__(self,value):super().__init__();self.value=float(value)
    def forward(self,obs,agent,source,weights):return torch.full((len(obs),5,5),self.value)

class PublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_legacy_module_cli_delegates_to_guarded_public_entry(self):
        from .paper_aligned_run import main
        with patch('repaired.public_run.main',return_value=17) as guarded:
            self.assertEqual(main(),17)
            guarded.assert_called_once_with()
    def test_lower_reuse_requires_seed_budget_world_and_sources(self):
        from .paper_aligned_run import load_or_train_lower
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model=ContextActionSF();sources=np.zeros((4,40,625,2),int)
            import hashlib
            payload={'model':model.state_dict(),'seed':100,'steps':520,'kappa':1.,'gamma':.95,
                'source_sha256':hashlib.sha256(sources.tobytes()).hexdigest()}
            torch.save(payload,root/'lower.pt');world=SFWorldExact(kappa=1.)
            load_or_train_lower(root,world,sources,[],100,520)
            for field,value in [('seed',101),('steps',640),('kappa',.5),('gamma',.9),('source_sha256','wrong')]:
                changed=dict(payload);changed[field]=value;torch.save(changed,root/'lower.pt')
                with self.assertRaisesRegex(ValueError,'Existing lower checkpoint differs'):
                    load_or_train_lower(root,world,sources,[],100,520)

    def test_scratch_exposes_only_direct_policy(self):
        from .scratch_learning import train_dqn, q_inputs
        with self.assertRaisesRegex(ValueError,'Only the direct'):
            train_dqn('/unused',None,100,'full',40,[],[],'fixture')
        states=np.array([0,1]);times=np.array([0,1]);weights=np.ones((2,2,4))
        x,base=q_inputs('direct',states,times,weights)
        self.assertEqual(tuple(x.shape),(2,13));self.assertIsNone(base)
        np.testing.assert_array_equal(x.numpy()[:,-8:],weights.reshape(2,8))
        with self.assertRaises(ValueError):q_inputs('source',states,times,weights)

    def test_blocking_is_explicit_reward_feature(self):
        world=SFWorldExact(kappa=1,horizon=2)
        w=np.zeros((2,4));feat=np.zeros((2,5));feat[0,4]=1
        self.assertEqual(world.reward(feat,w),-2.)
        # Agent 0 moves right into stationary agent 1 at cell 1.
        state=1;action=3*5+4
        self.assertEqual(world.expected_features[state,action,0,4],1.)
        self.assertEqual(world.reward(world.expected_features[state,action],w),-2.)
    def test_terminal_sf_target_does_not_bootstrap(self):
        world=SFWorldExact(horizon=2,kappa=.5)
        batch={'state':np.array([1]),'next_state':np.array([2]),'time':np.array([1]),
               'sources':np.array([[0,1]]),'weights':np.ones((1,2,4)),
               'action':np.array([3]),'features':np.ones((1,2,5)), 'done':np.ones(1)}
        sources=np.zeros((4,2,625,2),dtype=int)
        a=sf_batch_loss(ConstantSF(0),ConstantSF(0),batch,world,.95,sources)
        b=sf_batch_loss(ConstantSF(0),ConstantSF(100),batch,world,.95,sources)
        self.assertEqual(a.item(),b.item());self.assertEqual(a.item(),.5)
    def test_strict_entries_execute_only_source_actions(self):
        src=np.zeros((4,2,625,2),int)
        src[2,1,3,0]=4;src[1,1,3,1]=2
        self.assertEqual(_entry_to_primitive(src,2*4+1,1,3),4*5+2)
    def test_source_score_includes_blocking_feature(self):
        class BlockingOnly(ConstantSF):
            def forward(self,*args):
                out=super().forward(*args);out[...,4]=1;return out
        scores=candidate_values(BlockingOnly(0),np.array([0]),np.array([0]),np.ones((1,2,4)),np.zeros((4,40,625,2),int))
        np.testing.assert_array_equal(scores,-2*np.ones((1,2,4)))
    def test_training_loader_refuses_test(self):
        with self.assertRaises(PermissionError):load_tasks('/does/not/exist','test')
    def test_test_flag_required_before_any_objective_read(self):
        args=argparse.Namespace(output=Path('/does/not/exist'),allow_test=False)
        with patch('repaired.public_run.load_tasks',side_effect=AssertionError('must not read')):
            with self.assertRaises(PermissionError):evaluate_test(args)
    def test_smoke_cannot_be_frozen_as_paper_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);new_json(root/'completed.json',{'status':'COMPLETE','quick_smoke_only':True})
            with self.assertRaisesRegex(ValueError,'Quick smoke'):freeze(argparse.Namespace(output=root))
    def test_freeze_binds_artifacts_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);protocol=root/'protocol';create_protocol(protocol,seed=49831)
            identity=protocol_identity(protocol);src_hash='fixture'
            lower=ContextActionSF();upper=PaperAlignedAttention()
            torch.save({'model':lower.state_dict(),'seed':100,'steps':520,'updates':3,'kappa':1.,'gamma':.95,'source_sha256':src_hash},root/'lower.pt')
            torch.save({'model':upper.state_dict(),'seed':100,'budget':80,'mode':'ddqn','sources_sha256':src_hash},root/'upper.pt')
            new_json(root/'unit_binding.json',{'kappa':1.,'seed':100,'mode':'ddqn',
                'budget_upper':80,'budget_lower':520,'upper_sha256':sha(root/'upper.pt'),
                'lower_sha256':sha(root/'lower.pt'),'source_array_sha256':src_hash,
                'objective_protocol':identity})
            new_json(root/'summary.json',{'steps':80,'updates':1})
            new_json(root/'run_protocol.json',{'status':'fixture'})
            new_json(root/'evaluation_validation.json',{'status':'fixture'})
            new_json(root/'completed.json',{'status':'COMPLETE','quick_smoke_only':False,
                'objective_protocol':identity,'optimizer':'ddqn','runs':[{'seed':100,'kappa':1.,'mode':'ddqn',
                'budget_upper':80,'budget_lower':520,'upper':'upper.pt','lower':'lower.pt','binding':'unit_binding.json'}]})
            args=argparse.Namespace(output=root,protocol=protocol,allow_test=True)
            binding_path=root/'unit_binding.json';binding=json.loads(binding_path.read_text())
            for field,value in [('kappa',.5),('upper_sha256','wrong'),('source_array_sha256','wrong')]:
                changed=dict(binding);changed[field]=value;binding_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):freeze(args)
                self.assertFalse((root/'frozen.json').exists())
            binding_path.write_text(json.dumps(binding))
            freeze(args)
            with (root/'upper.pt').open('ab') as f:f.write(b'tamper')
            with patch('repaired.public_run.load_tasks',side_effect=AssertionError('must not read after tamper')):
                with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):evaluate_test(args)
            self.assertFalse((root/'test_access_started.json').exists())
    def test_json_evidence_is_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'receipt.json';new_json(p,{'status':'FIRST'})
            with self.assertRaises(FileExistsError):new_json(p,{'status':'SECOND'})
            self.assertEqual(json.loads(p.read_text())['status'],'FIRST')

if __name__=='__main__':unittest.main()
