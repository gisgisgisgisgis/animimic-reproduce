import json
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from animimic.core import Cache, load_config
from animimic.video_backends import backend_name, generate_video, validate_video


class VideoBackendTests(unittest.TestCase):
    def test_legacy_config_keeps_wan(self):
        self.assertEqual(backend_name(load_config("configs/fox.yaml")["video"]), "wan")
        with patch("animimic.models.generate_video") as wan:
            generate_video("input.png", "output.mp4", {"prompt": "walk"})
            wan.assert_called_once_with("input.png", "output.mp4", {"prompt": "walk"})

    def test_ltx_routes_without_wan(self):
        with patch("animimic.video_backends.generate_ltx") as ltx:
            config = {"backend": "ltx", "prompt": "walk", "num_frames": 49}
            generate_video("input.png", "output.mp4", config)
            ltx.assert_called_once_with("input.png", "output.mp4", config)

    def test_temporal_and_spatial_constraints_are_backend_specific(self):
        self.assertEqual(validate_video({"num_frames": 5}), "wan")
        with self.assertRaises(ValueError):
            validate_video({"backend": "ltx", "num_frames": 5})
        with self.assertRaises(ValueError):
            validate_video({"backend": "ltx", "width": 500})
        with self.assertRaises(ValueError):
            validate_video({"backend": "unrecognized"})
        with self.assertRaises(ValueError):
            validate_video({"backend": "wan", "width": 500})
        self.assertEqual(validate_video(load_config("configs/fox-ltx.yaml")["video"]), "ltx")

    def test_wan_smaller_resolution_runs_isolated_worker_with_allocator_config(self):
        from animimic.models import generate_video as generate_wan
        with tempfile.TemporaryDirectory(prefix="wan worker ") as directory:
            root = Path(directory)
            (root/'generate.py').touch()
            output = root/'reference.mp4'
            config = dict(backend='wan', repo=str(root), prompt='lift hoof',
                          width=832, height=480, num_frames=81)
            with patch.dict(os.environ, {}, clear=True), \
                 patch('animimic.models.subprocess.run') as process:
                process.side_effect = lambda *args, **kwargs: output.write_bytes(b'video')
                generate_wan(root/'input.png', output, config)
            args, kwargs = process.call_args
            self.assertEqual(Path(args[0][1]).name, 'wan_worker.py')
            self.assertEqual(json.loads(kwargs['input']), config)
            self.assertEqual(kwargs['env']['PYTORCH_CUDA_ALLOC_CONF'], 'expandable_segments:True')
            self.assertTrue(kwargs['check'])

    def test_full_quality_cpu_export_keeps_master_generation_settings(self):
        from animimic.models import generate_video as generate_wan
        root = Path(__file__).resolve().parents[1]
        # Validate fixed settings independently of the local Git remote availability.
        for action in ('leftforeleg', 'tail', 'mouth'):
            base = load_config(root/f'configs/experiments/horse-textured-wan-{action}.yaml')
            config = load_config(root/f'configs/experiments/horse-textured-wan-{action}-normalized-rigid.yaml')
            self.assertEqual(config['video'], base['video'])
            video = config['video'].copy()
            self.assertTrue(video.pop('cpu_export'))
            self.assertFalse(video.pop('use_negative_prompt'))
            self.assertEqual(video['num_frames'], 121)
            self.assertNotIn('width', video)
            self.assertNotIn('height', video)
            self.assertNotIn('sample_shift', video)
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo/'generate.py').touch()
            output = repo/'output.mp4'
            config = dict(prompt='lift hoof', repo=str(repo), cpu_export=True, num_frames=121)
            with patch('animimic.models.subprocess.run') as process:
                process.side_effect = lambda *args, **kwargs: output.write_bytes(b'video')
                generate_wan(repo/'input.png', output, config)
            self.assertEqual(Path(process.call_args.args[0][1]).name, 'wan_worker.py')
            self.assertEqual(json.loads(process.call_args.kwargs['input']), config)

    def test_wan_worker_passes_pixel_budget_and_offloads_before_export(self):
        import torch
        from PIL import Image
        from animimic.wan_worker import generate
        factory = Mock()
        factory.return_value.generate.return_value = torch.zeros(3, 5, 32, 32)
        wan = ModuleType('wan')
        wan.WanTI2V = factory
        configs = ModuleType('wan.configs')
        configs.WAN_CONFIGS = {'ti2v-5B': SimpleNamespace(sample_shift=5., sample_steps=50,
                                                       sample_guide_scale=5.)}
        utils = ModuleType('wan.utils.utils')
        utils.save_video = Mock()
        modules = {'wan': wan, 'wan.configs': configs, 'wan.utils.utils': utils}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
             patch.object(sys, 'path', sys.path.copy()):
            root = Path(directory)
            source, output = root/'input.png', root/'reference.mp4'
            Image.new('RGB', (64, 32)).save(source)
            utils.save_video.side_effect = lambda **kwargs: output.write_bytes(b'video')
            config = dict(prompt='lift hoof', negative_prompt='camera cuts', width=832, height=480,
                          num_frames=81, fps=24, num_inference_steps=40, sample_shift=3.)
            generate(root, root, source, output, config)
            low_call = factory.return_value.generate.call_args.kwargs.copy()
            # A faithful comparison uses the upstream sampler defaults and negative prompt,
            # matching the old CLI which never forwarded the YAML negative_prompt.
            generate(root, root, source, output, dict(prompt='lift hoof', negative_prompt='camera cuts',
                     use_negative_prompt=False, num_frames=121))
            faithful = factory.return_value.generate.call_args.kwargs
            self.assertEqual(faithful['max_area'], 1280*704)
            self.assertEqual(faithful['frame_num'], 121)
            self.assertEqual(faithful['shift'], 5.)
            self.assertEqual(faithful['n_prompt'], '')
        kwargs = factory.call_args.kwargs
        self.assertTrue(kwargs['t5_cpu'] and kwargs['init_on_cpu'] and kwargs['convert_model_dtype'])
        kwargs = low_call
        self.assertEqual(kwargs['max_area'], 832*480)
        self.assertEqual(kwargs['frame_num'], 81)
        self.assertEqual(kwargs['sampling_steps'], 40)
        self.assertEqual(kwargs['shift'], 3.)
        self.assertEqual(kwargs['n_prompt'], 'camera cuts')
        self.assertTrue(kwargs['offload_model'])
        self.assertEqual(utils.save_video.call_args.kwargs['tensor'].device.type, 'cpu')
        self.assertEqual(utils.save_video.call_args.kwargs['fps'], 24)

    def test_different_backend_gets_separate_cache_and_identical_input_reuses(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def produce(path):
                calls.append(path)
                (path/"reference.mp4").write_bytes(b"saved reference")
                return ["reference.mp4"]
            cache = Cache(root)
            wan = cache.run("video", {"backend": "wan", "seed": 1}, produce)
            ltx = cache.run("video", {"backend": "ltx", "seed": 1}, produce)
            again = cache.run("video", {"backend": "ltx", "seed": 1}, produce)
            self.assertNotEqual(wan, ltx)
            self.assertEqual(ltx, again)
            self.assertEqual(len(calls), 2)
