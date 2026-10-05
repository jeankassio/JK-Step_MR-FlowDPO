"""Meaningful CPU checks for preference construction and tensor integrity."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from jk_step.pairs import build_preference_pairs, select_mrsd_pairs, validate_pairs, _split_pairs
from jk_step.preprocess import (PreferenceTensorDataset, collate_pairs, validate_tensor_pair, preprocess_pairs,
                                model_weights_complete, preprocessing_models_ready)
from jk_step.rewards import semantic_consistency_reward, create_score_template


class RewardPairTests(unittest.TestCase):
    def candidates(self):
        return [{"audio_path":f"{i}.wav", "caption":"Piano and voice", "lyrics":"One two three",
                 "group_id":"song", "scores":{"text_alignment":a,"production_quality":b}}
                for i,(a,b) in enumerate(((0,0),(1,1),(2,2),(3,3),(4,4)))]

    def test_mrsd_requires_dominance_on_every_axis(self):
        samples = self.candidates()
        samples.append({**samples[-1], "audio_path":"tradeoff.wav",
                        "scores":{"text_alignment":5,"production_quality":-1}})
        pairs, report = select_mrsd_pairs(samples,axes=["text_alignment","production_quality"],
            primary_margins={"text_alignment":2,"production_quality":2},
            secondary_margins={"text_alignment":.1,"production_quality":.1},
            minimum_quantile=0,maximum_quantile=1,balance_axes=True)
        self.assertTrue(pairs)
        self.assertFalse(any("tradeoff" in p["chosen"] for p in pairs))
        self.assertEqual(sum(p['primary_axis']=='text_alignment' for p in pairs),
                         sum(p['primary_axis']=='production_quality' for p in pairs))
        self.assertEqual(report["algorithm"],"MRSD")
        for pair in pairs:
            for axis in report["axes"]:
                self.assertGreater(pair['chosen_scores'][axis],pair['rejected_scores'][axis])

    def test_groups_with_different_lyrics_are_rejected(self):
        samples = self.candidates()
        samples[1]["lyrics"]="Different words"
        with self.assertRaisesRegex(ValueError,"mixes captions/lyrics"):
            select_mrsd_pairs(samples,axes=["text_alignment","production_quality"])

    def test_missing_semantic_scores_never_invented(self):
        with self.assertRaisesRegex(ValueError,"semantic_consistency"):
            select_mrsd_pairs(self.candidates())

    def test_constant_axes_cannot_create_false_dominance(self):
        samples = self.candidates()
        for sample in samples:
            sample['scores']['production_quality']=1
        pairs,_ = select_mrsd_pairs(samples,axes=['text_alignment','production_quality'],
            minimum_quantile=0,maximum_quantile=1)
        self.assertEqual(pairs,[])

    def test_protected_diction_allows_equal_perfect_words_and_rejects_regression(self):
        samples=self.candidates()
        for sample in samples:
            sample['scores']['lyric_fidelity']=1.0
        samples[-1]['scores']['lyric_fidelity']=.5
        pairs,_=select_mrsd_pairs(samples,axes=['text_alignment','production_quality'],
            primary_quantile=.2,secondary_quantile=.1,minimum_quantile=0,maximum_quantile=1,
            protected_axes={'lyric_fidelity':0},minimum_protected_scores={'lyric_fidelity':.95})
        self.assertTrue(pairs)
        self.assertFalse(any(p['chosen']=='4.wav' for p in pairs))
        self.assertTrue(all(p['chosen_scores']['lyric_fidelity']==1. for p in pairs))

    def test_artist_holdout_keeps_all_song_pairs_together(self):
        pairs = [{'group_id':f'song{i}','holdout_group':f'artist{i//2}'} for i in range(8)]
        _split_pairs(pairs,.25,42)
        groups={}
        for pair in pairs:
            groups.setdefault(pair['holdout_group'],set()).add(pair['split'])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))
        self.assertIn('validation',{p['split'] for p in pairs})

    def test_explicit_cross_source_artist_group_survives_pair_selection(self):
        samples = []
        for song, raw_artist, holdout in (
                ('hf_song', 'Tulio Borges', 'canonical-tulio'),
                ('mtg_song', 'artist_461498', 'canonical-tulio'),
                ('other_song', 'artist_other', 'other-artist')):
            for candidate, value in enumerate((0, 10)):
                samples.append({'audio_path':f'{song}_{candidate}.wav', 'group_id':song,
                    'caption':song, 'lyrics':song, 'artist_id':raw_artist, 'holdout_group':holdout,
                    'scores':{'text_alignment':value, 'production_quality':value}})
        pairs,_ = select_mrsd_pairs(samples, axes=['text_alignment','production_quality'],
            primary_margins={'text_alignment':1,'production_quality':1},
            secondary_margins={'text_alignment':1,'production_quality':1},
            minimum_quantile=0, maximum_quantile=1)
        _split_pairs(pairs, .5, 42)
        shared = [p for p in pairs if p['group_id'] in ('hf_song','mtg_song')]
        self.assertEqual({p['holdout_group'] for p in shared}, {'canonical-tulio'})
        self.assertEqual(len({p['split'] for p in shared}), 1)
        self.assertEqual({p['split'] for p in pairs}, {'train','validation'})

    def test_semantic_formula_rewards_confident_matching_centroids(self):
        centroids=torch.eye(3)
        confident=semantic_consistency_reward(torch.eye(3),centroids,temperature=.1)
        uncertain=semantic_consistency_reward(torch.ones(3,3),centroids,temperature=.1)
        self.assertGreater(confident,uncertain)
        self.assertLessEqual(confident,0)
        self.assertAlmostEqual(uncertain,-np.log(3),places=5)

    def test_semantic_rejects_mismatched_representation(self):
        with self.assertRaisesRegex(ValueError,'share D'):
            semantic_consistency_reward(torch.ones(3,2),torch.eye(3))


class AudioPairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.audio=self.root/'input';self.audio.mkdir()
        sr=24000
        t=np.arange(sr)/sr
        audio=(.15*np.sin(2*np.pi*440*t)+.15*np.sin(2*np.pi*8000*t)).astype('float32')
        sf.write(self.audio/'song.wav',np.stack((audio,audio),axis=1),sr,subtype='FLOAT')
        (self.audio/'song.caption.txt').write_text('Piano with a clear sung voice',encoding='utf-8')
        (self.audio/'song.lyrics.txt').write_text('Every word remains here',encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_degradation_preserves_duration_and_lyrics_without_turbo(self):
        report=build_preference_pairs(str(self.audio),str(self.root/'output'),mode='degraded',
            cutoff_hz=3000,validation_fraction=0)
        raw=json.loads(Path(report['manifest']).read_text())
        self.assertEqual(report['pairs'],1)
        pair=raw['pairs'][0]
        self.assertEqual(pair['lyrics'],'Every word remains here')
        left,sr=sf.read(pair['chosen']);right,r_sr=sf.read(pair['rejected'])
        self.assertEqual(left.shape,right.shape)
        self.assertEqual(sr,r_sr)
        frequencies=np.fft.rfftfreq(len(left),1/sr)
        high=frequencies>6000
        self.assertLess(np.square(np.abs(np.fft.rfft(right[:,0]))[high]).sum(),
                        np.square(np.abs(np.fft.rfft(left[:,0]))[high]).sum()*.02)
        self.assertEqual(pair['chosen_scores'],{})
        self.assertEqual(pair['metadata']['pair_source'],'synthetic_acoustic_degradation')

    def test_import_validation_detects_duration_mismatch(self):
        audio,sr=sf.read(self.audio/'song.wav')
        sf.write(self.root/'short.wav',audio[:len(audio)//2],sr)
        pair={'id':'x','group_id':'song','chosen':str(self.audio/'song.wav'),
              'rejected':str(self.root/'short.wav'),'caption':'Piano','lyrics':'Hello'}
        report=validate_pairs({'pairs':[pair]})
        self.assertFalse(report['valid'])
        self.assertTrue(any('durations differ' in e for e in report['errors']))

    def test_duplicate_audio_cannot_leak_across_splits(self):
        pair={'id':'a','group_id':'one','chosen':str(self.audio/'song.wav'),
              'rejected':str(self.audio/'song.wav'),'caption':'Piano','split':'train'}
        second={**pair,'id':'b','group_id':'two','split':'validation'}
        report=validate_pairs({'pairs':[pair,second]})
        self.assertTrue(any('Audio leakage' in e for e in report['errors']))

    def test_score_template_preserves_sidecars_and_leaves_scores_unknown(self):
        report=create_score_template(str(self.audio),str(self.root/'scores.json'))
        sample=json.loads(Path(report['manifest']).read_text())['samples'][0]
        self.assertEqual(sample['lyrics'],'Every word remains here')
        self.assertIsNone(sample['scores']['semantic_consistency'])

    def test_vocal_stem_requires_exact_alignment(self):
        with self.assertRaisesRegex(ValueError,'vocal_path'):
            build_preference_pairs(str(self.audio),str(self.root/'output'),mode='degraded',degradation='vocal_lowpass')


class TensorPairTests(unittest.TestCase):
    def sample(self,length=30):
        frame=torch.arange(length).float().unsqueeze(-1).expand(length,64)
        return {'chosen_latents':frame,'rejected_latents':frame+1000,
                'context_latents':torch.cat((frame+2000,torch.ones(length,64)),dim=-1),
                'attention_mask':torch.ones(length,dtype=torch.bool),
                'encoder_hidden_states':torch.ones(5,8),
                'encoder_attention_mask':torch.tensor([1,1,1,0,0],dtype=torch.bool),
                'metadata':{'duration':length/25,'patch_size':2}}

    def test_aligned_crop_applies_to_both_branches_and_context(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'pair.pt';torch.save(self.sample(),path)
            dataset=PreferenceTensorDataset({'pairs':[{'id':'x','group_id':'song','tensor_path':str(path)}]},
                                            max_latent_length=8,seed=42)
            starts=[]
            for epoch in range(6):
                dataset.set_epoch(epoch)
                sample=dataset[0]
                starts.append(sample['crop_start'])
                self.assertEqual(sample['chosen_latents'].shape,(8,64))
                self.assertTrue(torch.equal(sample['rejected_latents']-sample['chosen_latents'],torch.full((8,64),1000.)))
                self.assertTrue(torch.equal(sample['context_latents'][:,:64]-sample['chosen_latents'],torch.full((8,64),2000.)))
                self.assertEqual(sample['crop_start']%2,0)
            self.assertGreater(len(set(starts)),1)
            self.assertEqual(dataset[0]['crop_start'],dataset[0]['crop_start'])

    def test_collation_masks_padding_and_matches_all_audio_branches(self):
        left=self.sample(8);right=self.sample(12)
        result=collate_pairs([left,right],pad_to_multiple_of=8)
        self.assertEqual(result['chosen_latents'].shape,(2,16,64))
        self.assertEqual(result['context_latents'].shape,(2,16,128))
        self.assertEqual(result['attention_mask'][0].sum(),8)
        self.assertEqual(result['attention_mask'][1].sum(),12)
        self.assertFalse(result['attention_mask'][1,12:].any())

    def test_invalid_internal_mask_is_rejected(self):
        sample=self.sample()
        sample['encoder_attention_mask']=torch.tensor([1,0,1,0,0])
        with self.assertRaisesRegex(ValueError,'contiguous valid prefix'):
            validate_tensor_pair(sample)

    def test_nonfinite_cache_is_rejected(self):
        sample=self.sample();sample['rejected_latents'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            validate_tensor_pair(sample)


class RealPipelineContractTests(unittest.TestCase):
    """Mock only model weights: exercise actual pair orchestration and caching."""
    setUp = AudioPairTests.setUp
    tearDown = AudioPairTests.tearDown
    def test_shared_chosen_context_and_cache_invalidation(self):
        report=build_preference_pairs(str(self.audio),str(self.root/'pairs'),mode='degraded',
                                     cutoff_hz=3000,validation_fraction=0)
        class VAE(torch.nn.Module):
            dtype=torch.float32
            def __init__(self):
                super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1))
            def encode(self,audio):
                values=torch.nn.functional.adaptive_avg_pool1d(audio,25).mean(dim=1,keepdim=True)
                latent=values.expand(-1,64,-1)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:latent))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1))
                self.config=SimpleNamespace(patch_size=1);self.tokenize_calls=0
            def tokenize(self,chosen,silence,mask):
                self.tokenize_calls+=1
                return chosen,torch.zeros(1,chosen.shape[1],dtype=torch.long),mask
            def detokenize(self,quantized):
                return quantized+42
        model=Model()
        def encode_text(*args,**kwargs):
            return torch.ones(1,3,8),torch.ones(1,3)
        def encoder(*args,**kwargs):
            return torch.ones(1,6,8),torch.tensor([[1,1,1,1,0,0]])
        with patch('jk_engine.models.loader.load_vae',return_value=VAE()) as vae_load, \
             patch('jk_engine.models.loader.load_text_encoder',return_value=(None,None)), \
             patch('jk_engine.models.loader.load_silence_latent',return_value=torch.zeros(1,100,64)), \
             patch('jk_engine.models.loader.load_decoder_for_training',return_value=model), \
             patch('jk_engine.models.loader.unload_models'), \
             patch('jk_engine.vendor.preprocess_text.encode_text',side_effect=encode_text), \
             patch('jk_engine.vendor.preprocess_lyrics.encode_lyrics',side_effect=encode_text), \
             patch('jk_engine.vendor.preprocess_encoder.run_encoder',side_effect=encoder):
            options={'auto_download':False,'device':'cpu','precision':'fp32','storage_dtype':'float32'}
            output=self.root/'cached'
            result=preprocess_pairs(report['manifest'],str(self.root/'ckpt'),'xl-sft',str(output),**options)
            self.assertEqual(result['processed'],1)
            self.assertEqual(model.tokenize_calls,1)
            pair=json.loads(Path(result['manifest']).read_text())['pairs'][0]
            cache=torch.load(pair['tensor_path'],weights_only=True)
            self.assertTrue(torch.allclose(cache['context_latents'][:,:64],cache['chosen_latents']+42))
            self.assertEqual(cache['metadata']['conditioning_source'],'chosen_only')
            previous=cache['cache_key']
            result2=preprocess_pairs(report['manifest'],str(self.root/'ckpt'),'xl-sft',str(output),**options)
            self.assertEqual(result2['processed'],1)
            self.assertEqual(vae_load.call_count,1)
            self.assertEqual(model.tokenize_calls,1)
            raw=json.loads(Path(report['manifest']).read_text());raw['pairs'][0]['lyrics']='Updated exact words'
            Path(report['manifest']).write_text(json.dumps(raw))
            preprocess_pairs(report['manifest'],str(self.root/'ckpt'),'xl-sft',str(output),**options)
            self.assertEqual(vae_load.call_count,2)
            self.assertEqual(model.tokenize_calls,2)
            self.assertNotEqual(torch.load(pair['tensor_path'],weights_only=True)['cache_key'],previous)

    def test_config_only_interrupted_download_triggers_setup(self):
        report=build_preference_pairs(str(self.audio),str(self.root/'pairs'),mode='degraded',
                                     cutoff_hz=3000,validation_fraction=0)
        checkpoint=self.root/'checkpoint'
        for component in ('vae','Qwen3-Embedding-0.6B','acestep-v15-xl-sft'):
            folder=checkpoint/component;folder.mkdir(parents=True)
            (folder/'config.json').write_text('{}')
        self.assertFalse(preprocessing_models_ready(checkpoint))
        with patch('jk_step.models.ensure_models',side_effect=RuntimeError('setup requested')) as setup:
            with self.assertRaisesRegex(RuntimeError,'setup requested'):
                preprocess_pairs(report['manifest'],str(checkpoint),'xl-sft',str(self.root/'cache'),device='cpu')
            setup.assert_called_once()


class ModelAvailabilityTests(unittest.TestCase):
    def test_local_single_file_and_truncated_weight(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}')
            self.assertFalse(model_weights_complete(root))
            weight=root/'model.safetensors'
            save_file({'weight':torch.ones(2,3)},str(weight))
            self.assertTrue(model_weights_complete(root))
            data=weight.read_bytes();weight.write_bytes(data[:-2])
            self.assertFalse(model_weights_complete(root))

    def test_sharded_index_requires_every_complete_shard(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
                'a':'model-00001-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}}))
            save_file({'a':torch.ones(2)},str(root/'model-00001-of-00002.safetensors'))
            self.assertFalse(model_weights_complete(root))
            save_file({'b':torch.ones(2)},str(root/'model-00002-of-00002.safetensors'))
            self.assertTrue(model_weights_complete(root))

    def test_shard_index_cannot_escape_component_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'../elsewhere.safetensors'}}))
            self.assertFalse(model_weights_complete(root))


if __name__=='__main__':
    unittest.main()
