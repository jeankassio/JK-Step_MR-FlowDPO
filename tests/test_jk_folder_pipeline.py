"""CPU orchestration fixtures only; these mocks do not establish ASR/caption quality."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from jk_step import dataset_pipeline as dp
from jk_step.audio_annotation import parse_caption, validate_phrases, whisper_segments, LocalAudioAnnotator


class FixtureAnnotator:
    """Explicit fake backend: never downloads/loads a model or claims real audio QA."""
    phrases = [{"start": 0., "end": 12., "text": "Fixture words"}]
    fields = {"caption": "CPU fixture caption only", "genre": "fixture", "vocal_status": "singing"}
    def __init__(self, *args):
        self.origin = {"test_backend": "mock; not real annotation"}
    def transcribe(self, audio, sr):
        return self.phrases
    def caption(self, path):
        return self.fields
    def unload_asr(self):
        pass
    def close(self):
        pass


def write_source(path, seconds=12, caption="User caption", lyrics="User supplied words"):
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 8000
    audio = (.1 * np.sin(np.arange(seconds * sr) * (2 * np.pi * 440 / sr))).astype(np.float32)
    sf.write(path, audio, sr, subtype="FLOAT")
    path.with_suffix('.txt').write_text(f"caption: {caption}\nlyrics:\n{lyrics}\n", encoding="utf-8")
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, path.with_suffix('.txt'))}


def fixture_preprocess(**options):
    """Valid tiny CPU tensors simulate encoding; they are not real ACE results."""
    out = Path(options['output_dir']);out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(options['dataset_json']).read_text(encoding='utf-8'))
    for sample in data['samples']:
        path = out / (Path(sample['audio_path']).stem + '.pt')
        if not path.exists():
            torch.save({'target_latents':torch.zeros(4,64),'attention_mask':torch.ones(4),
                        'encoder_hidden_states':torch.zeros(3,8),'encoder_attention_mask':torch.ones(3),
                        'context_latents':torch.zeros(4,128),'metadata':{'fixture':True}},path)
    return {'processed':len(data['samples']),'failed':0,'total':len(data['samples']),'output_dir':str(out)}


@pytest.fixture
def backend():
    with patch.object(dp,'LocalAudioAnnotator',FixtureAnnotator), \
         patch('jk_step.preprocess.preprocessing_models_ready',return_value=True), \
         patch('jk_step.preprocess.model_weights_complete',return_value=True), \
         patch('jk_engine.data.preprocess.preprocess_audio_files',side_effect=fixture_preprocess):
        yield


def config(tmp_path, **extra):
    return {'audio_dir':str(tmp_path/'source'),'output_dir':str(tmp_path/'generated'),
            'checkpoint_dir':str(tmp_path/'checkpoints'),'allow_download':False,'device':'cpu',
            'content_mode':'vocal',**extra}


def fixture_pair_preprocess(manifest, checkpoint_dir, model_variant, output_dir, **options):
    """Simulate only model encoding; preference audio is really constructed."""
    from jk_step.pairs import read_manifest, write_manifest
    data, _ = read_manifest(manifest)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for row in data['pairs']:
        path = out / (row['id'] + '.pt')
        if not path.exists() or options.get('force'):
            torch.save({'chosen_latents': torch.zeros(4,64), 'rejected_latents': torch.ones(4,64),
                        'attention_mask': torch.ones(4), 'encoder_hidden_states': torch.zeros(3,8),
                        'encoder_attention_mask': torch.ones(3), 'context_latents': torch.zeros(4,128),
                        'metadata': {'fixture': True}}, path)
        row['tensor_path'] = str(path)
    return {'manifest': write_manifest(out/'pairs.preprocessed.json', data), 'errors': []}


def test_folder_flow_dpo_builds_both_branches_and_preserves_source(tmp_path, backend):
    before = write_source(tmp_path/'source'/'song.wav')
    settings = config(tmp_path, objective='flow_dpo', pair_options={
        'cutoff_hz': 2000, 'clip_threshold': .04})
    with patch('jk_step.preprocess.preprocess_pairs', side_effect=fixture_pair_preprocess):
        result = dp.prepare_folder(settings)
    assert result['ready'] and result['objective'] == 'flow_dpo'
    assert result['pairs'] == result['preprocessed'] == 3
    assert not result['dataset_manifest'] and not result['supervised_manifest']
    assert not (tmp_path/'generated'/'supervised_manifest.json').exists()
    rows = json.loads(Path(result['pairs_manifest']).read_text())['pairs']
    assert {p['metadata']['degradation'] for p in rows} == {'lowpass','noise','clipping'}
    assert {p['caption'] for p in rows} == {'User caption'}
    assert {p['lyrics'] for p in rows} == {'User supplied words'}
    assert len({p['group_id'] for p in rows}) == 1
    assert {p['split'] for p in rows} == {'train'}
    for row in rows:
        assert row['tensor_sha256'] == hashlib.sha256(Path(row['tensor_path']).read_bytes()).hexdigest()
        a, sr = sf.read(row['chosen']); b, br = sf.read(row['rejected'])
        assert sr == br and a.shape == b.shape and np.max(np.abs(a-b)) > 1e-5
    assert before == {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}


def test_preference_audio_cache_recovers_tampering(tmp_path, backend):
    write_source(tmp_path/'source'/'song.wav')
    settings = config(tmp_path, objective='flow_dpo', preprocess=False,
                      pair_options={'degradation':'noise'})
    first = dp.prepare_folder(settings)
    row = json.loads(Path(first['raw_pairs_manifest']).read_text())['pairs'][0]
    rejected = Path(row['rejected']); original = rejected.read_bytes()
    second = dp.prepare_folder(settings)
    assert first['pairs_manifest'] == second['pairs_manifest'] and rejected.read_bytes() == original
    rejected.write_bytes(b'corrupt cached comparison')
    third = dp.prepare_folder(settings)
    assert third['pairs'] == 1 and rejected.read_bytes() == original
    assert not third['ready'] and third['status'] == 'annotated'


def test_preference_tensor_checksum_forces_encoding_after_finite_corruption(tmp_path, backend):
    write_source(tmp_path/'source'/'song.wav')
    settings = config(tmp_path, objective='flow_dpo', pair_options={'degradation':'noise'})
    with patch('jk_step.preprocess.preprocess_pairs', side_effect=fixture_pair_preprocess) as encoder:
        first = dp.prepare_folder(settings)
        row = json.loads(Path(first['pairs_manifest']).read_text())['pairs'][0]
        p = Path(row['tensor_path']); tensor = torch.load(p, weights_only=True)
        tensor['chosen_latents'][0,0] = 999.; torch.save(tensor, p)
        second = dp.prepare_folder(settings)
        assert encoder.call_args.kwargs['force'] is True
        assert second['ready']
        assert torch.load(p, weights_only=True)['chosen_latents'][0,0].item() == 0


def test_one_holdout_retains_training_pairs_at_high_validation_fraction(tmp_path, backend):
    write_source(tmp_path/'source'/'song.wav')
    with patch('jk_step.preprocess.preprocess_pairs', side_effect=fixture_pair_preprocess):
        result = dp.prepare_folder(config(tmp_path, objective='flow_dpo', validation_fraction=.8,
                                         pair_options={'degradation':'noise'}))
    assert result['ready']
    rows = json.loads(Path(result['pairs_manifest']).read_text())['pairs']
    assert {row['split'] for row in rows} == {'train'}


def test_identical_preference_is_excluded_and_remains_in_report_on_reuse(tmp_path, backend):
    write_source(tmp_path/'source'/'song.wav')
    settings = config(tmp_path, objective='flow_dpo', preprocess=False,
                      pair_options={'degradation':'mixed','cutoff_hz':2000,'clip_threshold':1})
    first = dp.prepare_folder(settings); second = dp.prepare_folder(settings)
    assert first['pairs'] == second['pairs'] == 2
    for result in (first, second):
        report = json.loads(Path(result['report']).read_text())
        assert any('identical' in entry['reason'] for entry in report['quarantine'])


def test_cancel_during_preference_preprocessing_never_becomes_failed_or_ready(tmp_path, backend):
    write_source(tmp_path/'source'/'song.wav'); event = threading.Event()
    def cancelled(*args, **kwargs):
        event.set()
        raise ValueError('Stopped while loading model')
    with patch('jk_step.preprocess.preprocess_pairs', side_effect=cancelled):
        result = dp.prepare_folder(config(tmp_path, objective='flow_dpo',
                                  pair_options={'degradation':'noise'}), stop_event=event)
    assert result['cancelled'] and result['status'] == 'cancelled' and not result['ready']


def test_cancel_during_final_preference_validation_never_reports_ready(tmp_path, backend):
    from jk_step.pairs import validate_pairs
    write_source(tmp_path/'source'/'song.wav'); event = threading.Event()
    def cancel_on_validation(*args, **kwargs):
        report = validate_pairs(*args, **kwargs)
        if kwargs.get('check_tensors'):
            event.set()
        return report
    with patch('jk_step.preprocess.preprocess_pairs', side_effect=fixture_pair_preprocess), \
         patch('jk_step.pairs.validate_pairs', side_effect=cancel_on_validation):
        result = dp.prepare_folder(config(tmp_path, objective='flow_dpo',
                                  pair_options={'degradation':'noise'}), stop_event=event)
    assert result['cancelled'] and result['status'] == 'cancelled' and not result['ready']


def test_invalid_preference_tensors_do_not_enable_training(tmp_path, backend):
    write_source(tmp_path/'source'/'song.wav')
    def invalid(*args, **kwargs):
        result = fixture_pair_preprocess(*args, **kwargs)
        data = json.loads(Path(result['manifest']).read_text())
        for row in data['pairs']:
            p = Path(row['tensor_path']); tensor = torch.load(p, weights_only=True)
            tensor['rejected_latents'][0,0] = float('nan'); torch.save(tensor, p)
        return result
    with patch('jk_step.preprocess.preprocess_pairs', side_effect=invalid):
        result = dp.prepare_folder(config(tmp_path, objective='flow_dpo',
                                         pair_options={'degradation':'noise'}))
    assert not result['ready'] and result['preprocessed'] == 0
    report = json.loads(Path(result['report']).read_text())
    assert any('nonfinite' in entry['reason'] for entry in report['quarantine'])


@pytest.mark.parametrize('extra', [
    {'degradation':'unknown'}, {'variations_per_audio':0}, {'variations_per_audio':True},
    {'noise_snr_db':float('nan')}, {'context_mode':'random'}, {'misspelled_option':5}])
def test_preference_options_reject_invalid_settings(tmp_path, extra):
    (tmp_path/'source').mkdir()
    with pytest.raises(ValueError):
        dp.validate_options(config(tmp_path, objective='flow_dpo', pair_options=extra))


def test_originals_preserved_names_unique_and_split_before_repeats(tmp_path,backend):
    before={}
    for i in range(3):
        before.update(write_source(tmp_path/'source'/str(i)/'same.wav',caption=f'User {i}',lyrics=f'Words {i}'))
    # Distinct source bytes avoid presenting duplicated audio as independent holdout.
    p=tmp_path/'source'/'2'/'same.wav';a,sr=sf.read(p);sf.write(p,a*.9,sr,subtype='FLOAT')
    before[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dp.prepare_folder(config(tmp_path))
    assert result['ready'] and result['preprocessed']==3
    rows=json.loads(Path(result['supervised_manifest']).read_text())['samples']
    assert len({r['id'] for r in rows})==3
    assert len({Path(r['metadata']['audio_path']).name for r in rows})==3
    assert {r['lyrics'] for r in rows}=={'Words 0','Words 1','Words 2'}
    holdouts={}
    for row in rows:
        holdouts.setdefault(row['holdout_group'],set()).add(row['split'])
        assert row['tensor_sha256']==hashlib.sha256(Path(row['tensor_path']).read_bytes()).hexdigest()
    assert all(len(s)==1 for s in holdouts.values())
    assert {r['split'] for r in rows}=={'train','validation'}
    assert before=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}


def test_same_audio_aliases_remain_in_same_holdout(tmp_path,backend):
    write_source(tmp_path/'source'/'a.wav',lyrics='one')
    write_source(tmp_path/'source'/'b.wav',lyrics='two')
    # Floating WAV PEAK metadata has a timestamp; force distinct container
    # headers deterministically instead of relying on whether a second elapses.
    first = (tmp_path/'source'/'a.wav').read_bytes()
    second = bytearray(first); timestamp = first.index(b'PEAK') + 12
    second[timestamp] ^= 1
    (tmp_path/'source'/'b.wav').write_bytes(second)
    assert first != bytes(second)
    assert np.array_equal(sf.read(tmp_path/'source'/'a.wav')[0], sf.read(tmp_path/'source'/'b.wav')[0])
    result=dp.prepare_folder(config(tmp_path))
    rows=json.loads(Path(result['supervised_manifest']).read_text())['samples']
    assert len({r['holdout_group'] for r in rows})==1
    assert {r['split'] for r in rows}=={'train'}


def test_resume_reuses_bytes_and_caption_edit_invalidates_tensor_directory(tmp_path,backend):
    write_source(tmp_path/'source'/'song.wav')
    first=dp.prepare_folder(config(tmp_path));second=dp.prepare_folder(config(tmp_path))
    assert first['tensor_dir']==second['tensor_dir'] and second['ready']
    p=tmp_path/'source'/'song.txt';p.write_text('caption: Changed by user\nlyrics:\nUser supplied words\n')
    third=dp.prepare_folder(config(tmp_path))
    assert third['tensor_dir']!=first['tensor_dir']


def test_human_lyrics_conflict_quarantines_without_instrumental_fallback(tmp_path,backend):
    before=write_source(tmp_path/'source'/'long.wav',seconds=40,lyrics='Human exact text')
    result=dp.prepare_folder(config(tmp_path))
    assert not result['ready'] and result['samples']==0 and result['status']=='failed'
    report=json.loads(Path(result['report']).read_text())
    assert any('disagree' in q['reason'] for q in report['quarantine'])
    assert before=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}


def test_unknown_auto_voice_cannot_be_declared_instrumental(tmp_path,backend):
    write_source(tmp_path/'source'/'song.wav',caption='',lyrics='')
    with patch.object(FixtureAnnotator,'phrases',[]),patch.object(FixtureAnnotator,'fields',{
            'caption':'CPU fixture caption','vocal_status':'uncertain'}):
        result=dp.prepare_folder(config(tmp_path,content_mode='auto'))
    assert not result['ready'] and result['samples']==0


def test_bad_tensor_not_reported_as_ready(tmp_path,backend):
    write_source(tmp_path/'source'/'song.wav')
    def bad(**options):
        result=fixture_preprocess(**options)
        for p in Path(options['output_dir']).glob('*.pt'):
            data=torch.load(p,weights_only=True);data['attention_mask'].zero_();torch.save(data,p)
        return result
    with patch('jk_engine.data.preprocess.preprocess_audio_files',side_effect=bad):
        result=dp.prepare_folder(config(tmp_path))
    assert not result['ready'] and result['preprocessed']==0


def test_cancel_is_never_ready(tmp_path,backend):
    write_source(tmp_path/'source'/'song.wav');event=threading.Event();event.set()
    result=dp.prepare_folder(config(tmp_path),stop_event=event)
    assert result['cancelled'] and result['status']=='cancelled' and not result['ready']


def test_input_output_overlap_rejected(tmp_path):
    (tmp_path/'source').mkdir()
    with pytest.raises(ValueError,match='outside'):
        dp.validate_options(config(tmp_path,output_dir=str(tmp_path/'source'/'generated')))


def test_phrase_boundaries_and_human_text_are_not_guessed():
    with pytest.raises(ValueError,match='incomplete'):
        validate_phrases([{'text':'word','timestamp':(0,None)}],20)
    with pytest.raises(ValueError,match='overlapping'):
        validate_phrases([{'text':'a','timestamp':(0,10)},{'text':'b','timestamp':(9,12)}],20)
    original="AÇÃO d’água!\nMeu amor."
    aligned=dp.preserve_human_lyrics([{'start':0,'end':10,'text':"Acao d'agua"},{'start':10,'end':20,'text':'Meu amor'}],original)
    assert [p['text'] for p in aligned]==['AÇÃO d’água!','Meu amor.']
    groups,bad=dp.phrase_groups([{'start':0,'end':31,'text':'full phrase'}],40,30)
    assert groups==[] and bad
    clips=dp._instrumental_clips(61,30.5)
    assert clips[0]['end']==clips[1]['start']==30.5


def test_caption_requires_audio_classification_json():
    with pytest.raises(ValueError):parse_caption('a filename-derived fallback')
    result=parse_caption('{"caption":"Clear fixture description","vocal_status":"singing","bpm":999}')
    assert result['bpm'] is None
    assert parse_caption("{'caption':'Quoted fixture description','vocal_status':'no_vocals'}")['vocal_status']=='no_vocals'
    with pytest.raises(ValueError,match='string'):
        parse_caption('{"caption":123,"vocal_status":"singing"}')
    with pytest.raises(ValueError):
        parse_caption("{'caption':__import__('os').getcwd(),'vocal_status':'singing'}")


def test_inspection_has_no_torch_import_or_download(tmp_path):
    write_source(tmp_path/'source'/'song.wav')
    code="import sys;from jk_step.dataset_pipeline import inspect_folder;assert 'torch' not in sys.modules;r=inspect_folder(sys.argv[1]);assert r['count']==1;assert 'torch' not in sys.modules;print('PASS')"
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'source')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('minimum',[0,float('nan'),float('inf'),31])
def test_minimum_clip_duration_is_validated(tmp_path,minimum):
    (tmp_path/'source').mkdir()
    with pytest.raises(ValueError,match='min_clip_seconds'):
        dp.validate_options(config(tmp_path,min_clip_seconds=minimum))


def test_minimum_clip_duration_controls_selection_and_cache(tmp_path,backend):
    write_source(tmp_path/'source'/'short.wav',seconds=2)
    first=dp.prepare_folder(config(tmp_path))
    assert not first['ready'] and first['samples']==0
    second=dp.prepare_folder(config(tmp_path,min_clip_seconds=1))
    assert second['ready'] and second['samples']==1
    clips,bad=dp.phrase_groups([{'start':0.,'end':2.,'text':'whole phrase'}],2,30,1)
    assert len(clips)==1 and not bad
    assert dp._instrumental_clips(2,30,1)[0]['end']==2
    third=dp.prepare_folder(config(tmp_path,min_clip_seconds=1.5))
    assert third['tensor_dir']!=second['tensor_dir']


def test_same_named_source_folders_have_separate_default_outputs(tmp_path):
    a=tmp_path/'one'/'music';b=tmp_path/'two'/'music'
    a.mkdir(parents=True);b.mkdir(parents=True)
    first=dp.validate_options({'audio_dir':str(a)})['output_dir']
    second=dp.validate_options({'audio_dir':str(b)})['output_dir']
    assert first!=second
    assert first==dp.validate_options({'audio_dir':str(a)})['output_dir']


def test_preprocess_only_sees_selected_inputs_not_stale_or_quarantined_clips(tmp_path,backend):
    write_source(tmp_path/'source'/'good.wav',lyrics='accepted words')
    write_source(tmp_path/'source'/'bad.wav',caption='',lyrics='')
    old=tmp_path/'generated'/'audio'/'old.flac';old.parent.mkdir(parents=True)
    sf.write(old,np.zeros(16000),8000)
    observed=[]
    def checked(**options):
        folder=Path(options['audio_dir'])
        data=json.loads(Path(options['dataset_json']).read_text())
        actual={p.name for p in folder.iterdir() if p.suffix in dp.AUDIO_SUFFIXES}
        expected={s['filename'] for s in data['samples']}
        assert actual==expected and len(actual)==1
        assert all(Path(s['audio_path']).parent==folder for s in data['samples'])
        assert folder.parent.name=='selected_inputs'
        observed.append(folder)
        return fixture_preprocess(**options)
    with patch.object(FixtureAnnotator,'fields',{'caption':'Mock ambiguous caption','vocal_status':'uncertain'}), \
         patch('jk_engine.data.preprocess.preprocess_audio_files',side_effect=checked):
        result=dp.prepare_folder(config(tmp_path,content_mode='auto'))
    assert result['ready'] and result['quarantined']==1 and len(observed)==1
    assert old.exists()
    rows=json.loads(Path(result['supervised_manifest']).read_text())['samples']
    assert all(Path(r['metadata']['audio_path']).parent==tmp_path/'generated'/'audio' for r in rows)


def test_native_whisper_requires_predicted_boundaries():
    class Tokenizer:
        def decode(self,tokens,**kwargs):
            assert tokens==[1,2]
            return 'Native fixture words'
    segment={'start':0.,'end':10.,'tokens':torch.tensor([100,1,2,600])}
    result=whisper_segments({'segments':[[segment]]},Tokenizer(),100,12)
    assert result==[{'start':0.,'end':10.,'text':'Native fixture words'}]
    with pytest.raises(ValueError,match='window-end'):
        whisper_segments({'segments':[[{**segment,'tokens':torch.tensor([100,1,2])}]]},Tokenizer(),100,12)
    with pytest.raises(ValueError,match='out-of-range'):
        whisper_segments({'segments':[[{**segment,'end':13.}]]},Tokenizer(),100,12)


def test_selected_input_manifest_tampering_is_not_silently_reused(tmp_path,backend):
    write_source(tmp_path/'source'/'good.wav')
    result=dp.prepare_folder(config(tmp_path))
    path=Path(result['selected_input_dir'])/'dataset.json'
    data=json.loads(path.read_text())
    data['samples'][0]['lyrics']='Changed after selection'
    path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='approved samples'):
        dp.prepare_folder(config(tmp_path))


def test_native_whisper_array_processing_retains_long_form_and_cooperative_stop(tmp_path):
    from types import SimpleNamespace
    calls={}
    event=threading.Event()
    class Processor:
        tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:'Native fixture words')
        def __call__(self,audio,**kwargs):
            calls['processor']=kwargs
            calls['length']=len(audio)
            return SimpleNamespace(input_features=torch.zeros(1,128,3500),attention_mask=torch.ones(1,3500))
    class Model:
        generation_config=SimpleNamespace(no_timestamps_token_id=99)
        def generate(self,**kwargs):
            calls['generate']=kwargs
            kwargs['monitor_progress'](torch.tensor([[0,3500]]))
            return {'segments':[[{'start':1.,'end':11.,'tokens':[150,1,2,650]}]]}
    annotator=LocalAudioAnnotator(dp.DEFAULTS|{'device':'cpu','allow_download':False},stop_event=event)
    annotator.device='cpu';annotator.asr=(Model(),Processor(),torch.float32)
    phrases=annotator.transcribe(np.zeros((35*16000,2),np.float32),16000)
    assert phrases[0]['start']==1 and phrases[0]['end']==11
    assert calls['length']==35*16000
    assert calls['processor']['truncation'] is False and calls['processor']['return_attention_mask'] is True
    assert calls['generate']['return_segments'] and calls['generate']['return_timestamps']
    event.set()
    assert calls['generate']['stopping_criteria'][0](None,None)
