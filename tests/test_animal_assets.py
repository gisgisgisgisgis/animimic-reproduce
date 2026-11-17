import unittest

from animimic.cli import ASSETS, ASSET_HASHES, ANIMAL_ASSETS
from animimic.core import load_config


class AnimalAssetsTests(unittest.TestCase):
    def test_eight_assets_with_five_distinct_documented_animal_sources(self):
        self.assertEqual(len(ASSETS), 8)
        self.assertEqual(set(ANIMAL_ASSETS), {"horse", "puppy", "shark", "seagull", "sea_turtle"})
        for name, spec in ANIMAL_ASSETS.items():
            self.assertGreater(spec["faces"], 4000)
            self.assertGreater(spec["joints"], 1)
            self.assertEqual(len(ASSET_HASHES[name]), 64)
            self.assertTrue(spec["source"].startswith("https://sketchfab.com/"))
            self.assertEqual(spec["license"], "CC BY 4.0")
            config = load_config(f"configs/{name}-ltx.yaml")
            self.assertEqual(config["asset"], name)
            self.assertEqual(config["video"]["backend"], "ltx")
