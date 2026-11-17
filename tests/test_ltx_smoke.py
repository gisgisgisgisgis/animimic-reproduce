import importlib.util
import unittest

import torch


@unittest.skipUnless(importlib.util.find_spec("diffusers") and importlib.util.find_spec("transformers"),
                     "Install the video extra for offline LTX smoke test")
class LTXSmokeTests(unittest.TestCase):
    def test_real_diffusers_image_to_video_without_downloading_weights(self):
        from PIL import Image
        from diffusers import (AutoencoderKLLTXVideo, LTXVideoTransformer3DModel,
                               FlowMatchEulerDiscreteScheduler, LTXImageToVideoPipeline)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            vae = AutoencoderKLLTXVideo(
                latent_channels=4, block_out_channels=(32, 32, 32, 32),
                decoder_block_out_channels=(32, 32, 32, 32),
                layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1, 1))
            transformer = LTXVideoTransformer3DModel(
                in_channels=4, out_channels=4, num_attention_heads=2,
                attention_head_dim=8, cross_attention_dim=16, num_layers=1, caption_channels=16)
            pipe = LTXImageToVideoPipeline(
                vae=vae, transformer=transformer, tokenizer=None, text_encoder=None,
                scheduler=FlowMatchEulerDiscreteScheduler())
            pipe.set_progress_bar_config(disable=True)
            with torch.inference_mode():
                result = pipe(image=Image.new("RGB", (32, 32), "gray"),
                    height=32, width=32, num_frames=9, frame_rate=24,
                    num_inference_steps=1, guidance_scale=1,
                    prompt_embeds=torch.zeros(1, 2, 16),
                    prompt_attention_mask=torch.ones(1, 2, dtype=torch.long))
            self.assertEqual(len(result.frames[0]), 9)
            self.assertEqual(result.frames[0][0].size, (32, 32))
        finally:
            torch.set_num_threads(previous_threads)
