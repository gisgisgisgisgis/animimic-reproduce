"""Compare adapted and pinned reference Stage 1 code on identical cached inputs.

The reference loop and losses are bundled unchanged at a pinned commit. Only its camera,
renderer and FK/LBS dependencies are adapted to the shared scene representation.
No video generation, tracking, rigging, ARAP or FEM is run here.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from .core import Cache, digest, file_hash, save_history, save_json
from .optimize import fit_joints, tensors
from .rig import fk, lbs, pose_local
from .reference_stage1 import COMMIT as REFERENCE_COMMIT, SOURCE_HASHES


class SharedCamera:
    def __init__(self, renderer):
        self.renderer = renderer
        self.image_size = max(renderer.size)

    def project(self, points):
        if points.ndim == 3:
            return (torch.stack([self.renderer.project(p) for p in points]),
                    torch.stack([self.renderer.depth(p) for p in points]))
        return self.renderer.project(points), self.renderer.depth(points)


class SharedRenderer:
    has_rgb = True

    def __init__(self, rig, renderer):
        self.rig, self.renderer = rig, renderer

    def __call__(self, vertices):
        images, masks = zip(*(self.renderer(v, self.rig['faces'], self.rig['colors']) for v in vertices))
        return torch.stack(images), torch.stack(masks)


class SeededTorch:
    """Give the reference loop the same CPU-generated frame permutations as fit_joints."""
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def randperm(self, count, device=None):
        return torch.randperm(count, generator=self.generator).to(device=device)

    def __getattr__(self, name):
        return getattr(torch, name)


def load_reference(cache):
    commit = REFERENCE_COMMIT
    bundled = Path(__file__).with_name('reference_stage1')
    directory = Path(cache)/commit
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ('rig', 'losses', 'stage1'):
        source = bundled/f'{name}.py'
        if not source.is_file():
            raise FileNotFoundError(f'Missing bundled reference source: {source}. Sync the entire reference_stage1 directory.')
        if file_hash(source) != SOURCE_HASHES[name]:
            raise ValueError(f'Bundled reference source was modified: {source}')
        content = source.read_bytes()
        path = directory/f'{name}.py'
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
        hashes[name] = file_hash(path)
    package = f'animimic_reference_{commit[:12]}'
    module = ModuleType(package)
    module.__path__ = [str(directory)]
    sys.modules[package] = module
    # These modules provide type annotations and dependency injection only.
    # The actual reference rig/loss/Stage 1 files are imported without rewriting.
    for name, values in (
        ('camera', {'Camera': SharedCamera}),
        ('render', {'build_renderer': None}),
        ('sequence', {'Sequence': object}),
    ):
        stub = ModuleType(f'{package}.{name}')
        stub.__dict__.update(values)
        sys.modules[stub.__name__] = stub
    spec = importlib.util.spec_from_file_location(f'{package}.stage1', directory/'stage1.py')
    stage1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage1
    spec.loader.exec_module(stage1)
    return stage1, {'commit': commit, 'source_hashes': hashes}


def run_reference(stage1, rig, target, renderer, output, config):
    stage1.build_renderer = lambda *args: SharedRenderer(rig, renderer)
    stage1.forward_kinematics = lambda unused, rotation, translation: (
        fk(pose_local(rig['local'], rotation, translation), rig['parents'].tolist()), None)
    stage1.linear_blend_skinning = lambda unused, vertices, world, unused_translation: lbs(
        vertices, rig['weights'], world, rig['inverse_bind'])
    stage1.torch = SeededTorch(config.get('seed', 0))
    h, w = renderer.size
    tracks = target['tracks']
    valid = ((tracks[..., 0] >= 0) & (tracks[..., 0] <= w-1)
             & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h-1))
    visible = (target['confidence'] > config.get('confidence_threshold', .1)).clone()
    visible[0] = True
    visible &= valid
    seq = SimpleNamespace(
        num_frames=len(target['frames']), mesh=SimpleNamespace(verts=rig['vertices'], faces=rig['faces']),
        rig=SimpleNamespace(num_joints=len(rig['parents']), parents=rig['parents'].tolist()),
        camera=SharedCamera(renderer), mask=target['masks'],
        rgb=target['frames']*target['masks'][..., None]+1-target['masks'][..., None],
        tracks=SimpleNamespace(face=target['triangles'], bary=target['barycentric'],
                               xy=tracks, visible=visible, depth=target['depth']))
    weights = config['loss_weights']
    cfg = stage1.Stage1Config(
        iters=config['iterations'], lr=config.get('learning_rate', .001),
        batch_frames=config['batch_frames'], depth_norm='per_frame', depth_mode='points',
        rot_max=config.get('rotation_bound', 1.), trans_max=config.get('translation_bound', .01),
        **{f'w_{dest}': weights[src] for src, dest in
           [('rgb', 'rgb'), ('mask', 'mask'), ('track', 'track'), ('depth', 'depth'),
            ('smooth', 'smooth'), ('regularization', 'reg'), ('rigid', 'rigid')]})
    result = stage1.run_stage1(seq, cfg, rig['vertices'].device, backend='shared')
    rotation, translation = result['r6d'], result['t']
    local = pose_local(rig['local'], rotation, translation)
    world = fk(local, rig['parents'].tolist())
    vertices = lbs(rig['vertices'], rig['weights'], world, rig['inverse_bind'])
    if not torch.isfinite(vertices).all():
        raise RuntimeError('Reference optimizer returned nonfinite vertices; inspect track/depth validity')
    np.savez_compressed(output/'joint_trajectory.npz', vertices=vertices.detach().cpu(),
                        local=local.detach().cpu(), world=world.detach().cpu(),
                        rotation=rotation.cpu(), translation=translation.cpu())
    save_history(output/'joint_losses.json', result['history'])
    return ['joint_trajectory.npz', 'joint_losses.json', 'joint_losses.csv']


def artifact(manifest, stage, name, repository):
    path = Path(manifest['stages'][stage]['directory'])/name
    if not path.is_absolute():
        path = repository/path
    expected = manifest['stages'][stage]['artifacts'][name]
    if not path.is_file():
        # Transfers can nest the cache root; accept only the same key and hash.
        matches = (repository/'.cache').glob(f'**/{stage}/{path.parent.name}/{name}')
        path = next((p for p in matches if file_hash(p) == expected), path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing cached input: {path}. Run this command on the machine holding this run cache.')
    if file_hash(path) != expected:
        raise ValueError(f'Cached input hash does not match this run: {path}')
    return path


def compare(run_directory, device='cuda', iterations=None, batch_frames=None, previews=True):
    from .evaluate import metrics, deformation_metrics
    from .joint_review import source_fps, write_joint_review
    from .render import Renderer
    from .optimize import objective_weights

    repository = Path(__file__).resolve().parents[2]
    run_directory = Path(run_directory).resolve()
    manifest = json.loads((run_directory/'manifest.json').read_text())
    # Require the exact saved observations; never regenerate or substitute another run.
    inputs = {stage: artifact(manifest, stage, name, repository) for stage, name in
              [('prepare', 'rig.npz'), ('supervision', 'supervision.npz')]}
    video = run_directory/'reference.mp4'
    if not video.is_file() or file_hash(video) != manifest['stages']['video']['artifacts']['reference.mp4']:
        raise ValueError('reference.mp4 is missing or does not match this run')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run on the GPU host, or use --device cpu for a small smoke test')
    config = {**manifest['config']['joints'], 'objective': 'normalized'}
    config['iterations'] = iterations if iterations is not None else config.get('iterations', 3000)
    config['batch_frames'] = batch_frames if batch_frames is not None else config.get('batch_frames', 1)
    if min(config['iterations'], config['batch_frames']) < 1:
        raise ValueError('iterations and batch_frames must be positive')
    if config.get('optimize_joint_translations', True) is False:
        raise ValueError('Comparison requires joint translations enabled for both optimizers')
    config['loss_weights'] = objective_weights(config)
    if any(config['loss_weights'][k] for k in ('stationary', 'rotation_smooth', 'root_smooth')):
        raise ValueError('Disable experimental priors not present in the reference optimizer')
    reference, provenance = load_reference(repository/'.cache/optimizer-reference')
    source = Path(__file__).parent
    record = {'reference': provenance, 'settings': config, 'camera': manifest['config']['camera'],
              'inputs': {k: file_hash(v) for k, v in inputs.items()}, 'video': file_hash(video),
              'device': str(device), 'torch': torch.__version__,
              'code': {name: file_hash(source/name) for name in
                       ('compare_optimizers.py', 'optimize.py', 'losses.py', 'rig.py', 'render.py', 'evaluate.py')},
              'shared': ['rig', 'FK/LBS', 'camera', 'renderer', 'frames', 'tracks', 'depth',
                         'visibility mask', 'initial pose', 'frame permutations', 'settings'],
              'differences': ['reference edge counting and short-edge targets',
                              'reference normalization behavior for sparse visible tracks']}
    output = run_directory/'optimizer-comparison'/digest(record)[:12]
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'inputs.json', record)
    rig = tensors(dict(np.load(inputs['prepare'])), device)
    data = dict(np.load(inputs['supervision']))
    target = {k: torch.as_tensor(v, device=device, dtype=torch.long if k in ('triangles', 'indices')
                                else torch.float32) for k, v in data.items() if k != 'depth_maps'}
    fps = source_fps(video)
    target['frame_times'] = target['indices'].float()/fps
    renderer = Renderer(device, **manifest['config']['camera'])
    renderer.bind_material(rig)
    report = {'settings': config, 'reference': provenance, 'arms': {}}
    cache = Cache(output)
    for arm in ('adapted', 'reference'):
        print(f'[comparison] Running {arm} optimizer', flush=True)
        def produce(directory):
            torch.manual_seed(config.get('seed', 0))
            files = (fit_joints(rig, target, renderer, directory, config) if arm == 'adapted' else
                     run_reference(reference, rig, target, renderer, directory, config))
            result = dict(np.load(directory/'joint_trajectory.npz'))
            vertices = torch.as_tensor(result['vertices'], device=device)
            scores = {**metrics(vertices, rig, target, renderer),
                      **deformation_metrics(vertices, rig, target['frame_times'])}
            valid = target['confidence'] > config.get('confidence_threshold', .1)
            valid = valid.clone()
            valid[0] = True
            points = (vertices[:, rig['faces'][target['triangles']]]*target['barycentric'][None, ..., None]).sum(-2)
            projected = torch.stack([renderer.project(p) for p in points])
            tracks = target['tracks']
            h, w = renderer.size
            valid &= ((tracks[..., 0] >= 0) & (tracks[..., 0] <= w-1)
                      & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h-1))
            moving = (tracks-tracks[:1]).norm(dim=-1).masked_fill(~valid, 0).amax(0) > 2.
            selected = valid & moving[None]
            scores['moving_track_error_pixels'] = float((projected-tracks).norm(dim=-1)[selected].mean()) if selected.any() else None
            save_json(directory/'metrics.json', scores)
            return files+['metrics.json']
        directory = cache.run(arm, record, produce)
        result = dict(np.load(directory/'joint_trajectory.npz'))
        if previews:
            write_joint_review(directory/'joint-review', rig, result['local'], renderer,
                               video, data['indices'], fps)
        report['arms'][arm] = {'directory': str(directory),
                               'metrics': json.loads((directory/'metrics.json').read_text())}
    save_json(output/'comparison.json', report)
    print(f'[comparison] Results: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--batch-frames', type=int)
    parser.add_argument('--no-previews', action='store_true')
    args = parser.parse_args()
    compare(args.run_directory, args.device, args.iterations, args.batch_frames, not args.no_previews)


if __name__ == '__main__':
    main()
