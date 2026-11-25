import unittest

from animimic.cli import ASSETS, ASSET_HASHES, ANIMAL_ASSETS
from animimic.core import load_config


class AnimalAssetsTests(unittest.TestCase):
    def test_nine_assets_with_documented_animal_sources(self):
        self.assertEqual(len(ASSETS), 9)
        self.assertEqual(set(ANIMAL_ASSETS), {"horse", "horse_textured", "puppy", "shark", "seagull", "sea_turtle"})
        for name, spec in ANIMAL_ASSETS.items():
            self.assertGreater(spec["faces"], 1000)
            self.assertGreater(spec["joints"], 1)
            self.assertEqual(len(ASSET_HASHES[name]), 64)
            self.assertTrue(spec["source"].startswith("https://sketchfab.com/"))
            self.assertEqual(spec["license"], "CC BY 4.0")
            if "config" in spec:
                config = load_config(spec["config"])
                self.assertEqual(config["asset"], name)
                self.assertEqual(config["video"]["backend"], "wan")
