import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from animimic.supervision_review import summarize, write_review


class SupervisionReviewTests(unittest.TestCase):
    def data(self):
        masks = np.ones((3, 8, 10))
        masks[1:, :, 5:] = 0
        return dict(frames=np.ones((3, 8, 10, 3)), masks=masks,
                    tracks=np.array([[[2., 2.], [8., 2.], [20., 0.]]]*3),
                    visibility=np.ones((3, 3)), confidence=np.ones((3, 3)),
                    depth=np.array([[1., 1.002, 100.]]*3), indices=np.array([0, 5, 10]))

    def test_reports_mask_track_conflict_and_small_depth_scale(self):
        data = self.data()
        report = summarize(data)
        first, last = report['per_frame'][0], report['per_frame'][-1]
        self.assertEqual(last['reliable_tracks'], 3)
        self.assertEqual(last['reliable_in_bounds'], 2)
        self.assertEqual(first['reliable_inside_mask'], 1.)
        self.assertEqual(last['reliable_inside_mask'], .5)
        self.assertEqual(last['displacement_median_px'], 0.)
        self.assertAlmostEqual(last['depth_mad'], .001)
        self.assertAlmostEqual(last['depth_mad_inverse'], 1000.)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'supervision.npz'
            np.savez(path, **data)
            write_review(path, Path(temporary)/'review')
            self.assertTrue((Path(temporary)/'review/supervision-review.csv').is_file())
            with Image.open(Path(temporary)/'review/supervision-review.jpg') as sheet:
                self.assertEqual(sheet.size, (20, 99))
        np.testing.assert_array_equal(data['masks'][1, :, 5:], 0)

    def test_rejects_corrupt_inputs_and_handles_no_reliable_tracks(self):
        for key in ('visibility', 'confidence', 'depth', 'tracks'):
            data = self.data()
            data[key].flat[0] = np.nan
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                summarize(data)
        data = self.data()
        data['visibility'][:] = 0
        last = summarize(data)['per_frame'][-1]
        self.assertIsNone(last['depth_mad'])
        self.assertIsNone(last['reliable_inside_mask'])

    def test_review_video_keeps_source_timeline_and_depth_panel(self):
        import imageio.v2 as iio

        data = self.data()
        data["depth_maps"] = np.broadcast_to(np.arange(10), data["masks"].shape)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output/"supervision.npz"
            np.savez(path, **data)
            report = write_review(path, output, source_fps=30.)
            self.assertEqual(list(output.glob("*.png")), [])
            with iio.get_reader(output/report["video"]) as video:
                self.assertEqual(video.count_frames(), 11)
                self.assertEqual(video.get_meta_data()["fps"], 30.)
                self.assertEqual(video.get_meta_data()["size"], (30, 32))
