"""Explicit synthetic preference preparation from annotated music clips.

These comparisons teach acoustic degradation avoidance. They are not MRSD
reward evaluations, and the original recordings are not certified high quality.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .audio_annotation import check_stop, read_audio


def preference_options(value=None):
    settings = dict(value or {})
    allowed = {'degradation', 'variations_per_audio', 'cutoff_hz', 'noise_snr_db',
               'clip_threshold', 'context_mode'}
    if set(settings) - allowed:
        raise ValueError('Unknown pair_options: ' + ', '.join(sorted(set(settings) - allowed)))
    settings.setdefault('degradation', 'mixed')
    if settings['degradation'] not in ('mixed', 'lowpass', 'noise', 'clipping'):
        raise ValueError('degradation must be mixed, lowpass, noise or clipping')
    settings.setdefault('variations_per_audio', 3 if settings['degradation'] == 'mixed' else 1)
    variations = settings['variations_per_audio']
    if isinstance(variations, bool) or not isinstance(variations, int) or not 1 <= variations <= 100:
        raise ValueError('variations_per_audio must be an integer between 1 and 100')
    for key, default, minimum, maximum in (
        ('cutoff_hz', 6000, 100, 20000), ('noise_snr_db', 24, 0, 80),
        ('clip_threshold', .15, .00001, 1)):
        number = float(settings.get(key, default))
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f'{key} must be between {minimum} and {maximum}')
        settings[key] = number
    settings.setdefault('context_mode', 'chosen_semantic')
    if settings['context_mode'] not in ('chosen_semantic', 'silence'):
        raise ValueError('context_mode must be chosen_semantic or silence')
    return settings


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_preferences(dataset_json, options, progress=None, stop_event=None):
    from .dataset_pipeline import _fingerprint, _write_json
    from .pairs import build_preference_pairs, read_manifest, validate_pairs, write_manifest, _split_pairs
    settings = preference_options(options.get('pair_options'))
    dataset = json.loads(Path(dataset_json).read_text(encoding='utf-8'))
    samples = [{**sample, 'group_id': sample['id']} for sample in dataset['samples']]
    key = _fingerprint({'version': 1, 'samples': samples, 'settings': settings,
                        'seed': options['seed'], 'validation_fraction': options['validation_fraction']})
    destination = Path(options['output_dir']) / 'preferences' / key[:24]
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / 'annotated_clips.json'
    _write_json(source, {**dataset, 'samples': samples})
    manifest = destination / 'pairs.json'
    seal = destination / 'pair_cache.json'
    def emit(stage, message, current=0, total=0):
        check_stop(stop_event)
        if progress:
            progress({'event': 'progress', 'stage': stage, 'message': message,
                      'current': current, 'total': total})
    check_stop(stop_event)
    reused = False
    saved = {}
    if options['reuse'] and manifest.is_file() and seal.is_file():
        try:
            saved = json.loads(seal.read_text(encoding='utf-8'))
            reused = saved['key'] == key and saved['manifest_sha256'] == _sha(manifest)
            reused = reused and all(_sha(path) == digest for path, digest in saved['audio_sha256'].items())
            reused = reused and validate_pairs(manifest, check_audio=True)['valid']
        except (OSError, ValueError, KeyError):
            reused = False
    if not reused:
        kinds = ['lowpass', 'noise', 'clipping'] if settings['degradation'] == 'mixed' else settings['degradation']
        emit('pairs', 'Creating controlled acoustic preference comparisons')
        build_preference_pairs(str(source), str(destination), mode='degraded',
            **{key: value for key, value in settings.items() if key not in ('degradation', 'context_mode')},
            degradation=kinds, seed=options['seed'], validation_fraction=options['validation_fraction'],
            stop_event=stop_event, progress_callback=lambda current,total,message: emit('pairs', message,current,total))
    raw, _ = read_manifest(manifest)
    previous_splits = [pair.get('split') for pair in raw['pairs']]
    _split_pairs(raw['pairs'], options['validation_fraction'], options['seed'])
    splits_changed = previous_splits != [pair['split'] for pair in raw['pairs']]
    accepted, exclusions = [], list(saved.get('exclusions', [])) if reused else []
    import numpy as np
    for pair in raw['pairs']:
        check_stop(stop_event)
        chosen, sr = read_audio(pair['chosen'])
        rejected, rejected_sr = read_audio(pair['rejected'])
        if sr != rejected_sr or chosen.shape != rejected.shape:
            raise ValueError('Synthetic pair changed sample alignment')
        rms = float(np.sqrt(np.mean(chosen ** 2)))
        difference = float(np.sqrt(np.mean((chosen - rejected) ** 2)))
        if rms <= 1e-8 or difference <= max(1e-7, rms * 1e-5):
            exclusions.append({'id': pair['id'], 'reason': 'Silent or effectively identical preference branches'})
            continue
        accepted.append(pair)
    if not accepted:
        raise ValueError('No nontrivial acoustic preference pairs were produced')
    if not reused or splits_changed or len(accepted) != len(raw['pairs']):
        raw['pairs'] = accepted
        write_manifest(manifest, raw)
        _write_json(seal, {'key': key, 'manifest_sha256': _sha(manifest),
            'exclusions': exclusions,
            'audio_sha256': {pair[branch]: _sha(pair[branch]) for pair in accepted for branch in ('chosen', 'rejected')}})
    check_stop(stop_event)
    if not options['preprocess']:
        return {'pairs_manifest': str(manifest), 'raw_pairs_manifest': str(manifest),
                'tensor_dir': '', 'pairs': len(accepted), 'rows': [], 'exclusions': exclusions}
    from .preprocess import preprocess_pairs
    emit('preprocess', 'Encoding both preference branches with shared caption and lyrics')
    tensors = destination / 'tensors'
    force = not options['reuse']
    previous = tensors / 'pairs.preprocessed.json'
    if options['reuse'] and previous.is_file():
        try:
            cached, _ = read_manifest(previous)
            force = any(pair.get('tensor_sha256') and
                        _sha(pair['tensor_path']) != pair['tensor_sha256']
                        for pair in cached['pairs'])
        except (OSError, ValueError, KeyError):
            force = True
    result = preprocess_pairs(str(manifest), options['checkpoint_dir'], options['model_variant'],
        str(tensors), auto_download=options['allow_download'], device=options['device'],
        precision=options['precision'], context_mode=settings['context_mode'], max_duration=0,
        normalize='none', custom_tag=options['custom_tag'], vae_posterior='mean', fail_fast=False,
        checksums=True,
        force=force, stop_event=stop_event,
        progress_callback=lambda current,total,message: emit('preprocess',message,current,total))
    check_stop(stop_event)
    processed, _ = read_manifest(result['manifest'])
    rows = processed['pairs']
    for pair in rows:
        pair['tensor_sha256'] = _sha(pair['tensor_path'])
    validation = validate_pairs(processed, check_audio=True, check_tensors=True)
    if rows and not validation['valid']:
        raise ValueError('Invalid prepared preferences: ' + '; '.join(validation['errors'][:5]))
    write_manifest(Path(result['manifest']), processed)
    exclusions.extend({'id': entry.get('id'), 'reason': entry.get('error', str(entry))}
                      for entry in result.get('errors', []))
    check_stop(stop_event)
    return {'pairs_manifest': result['manifest'], 'raw_pairs_manifest': str(manifest),
            'tensor_dir': str(tensors), 'pairs': len(accepted), 'rows': rows, 'exclusions': exclusions}
