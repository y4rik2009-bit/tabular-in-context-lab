import unittest

import numpy as np
import torch

from benchmark import make_split, predict_icl
from miniicl import MiniTabICL
from prior import sample_batch


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_query_labels_cannot_enter_forward(self):
        rng = np.random.default_rng(7)
        x, y, nf = sample_batch(rng, 2, 8, 2, 5)
        model = MiniTabICL(width=16)
        logits = model(x, y[:, :8], nf)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_query_independence_at_prediction(self):
        model = MiniTabICL(width=16)
        rng = np.random.default_rng(3)
        xc = rng.normal(size=(8, 4)).astype(np.float32)
        yc = np.array([0, 1] * 4)
        xe = rng.normal(size=(3, 4)).astype(np.float32)
        one = predict_icl(model, xc, yc, xe[:1], 5, batch_size=1)
        many = predict_icl(model, xc, yc, xe, 5, batch_size=3)
        np.testing.assert_allclose(one, many[:1], rtol=0, atol=1e-6)

    def test_split_disjoint_and_context_only_scaling(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(120, 6))
        y = np.tile([0, 1], 60)
        xc, yc, xe, ye, context, test = make_split(x, y, 9, 24, 20, 5)
        self.assertFalse(set(context) & set(test))
        self.assertEqual(xc.shape, (24, 5))
        self.assertEqual(xe.shape, (20, 5))
        self.assertEqual(len(yc), 24)
        self.assertEqual(len(ye), 20)

    def test_context_row_permutation_invariance(self):
        torch.manual_seed(0)
        model = MiniTabICL(width=16).eval()
        x, y, nf = sample_batch(np.random.default_rng(12), 1, 8, 1, 5)
        order = torch.tensor([4, 2, 7, 0, 5, 1, 6, 3])
        with torch.inference_mode():
            a = model(x, y[:, :8], nf)
            b = model(torch.cat((x[:, order], x[:, 8:]), dim=1),
                      y[:, order], nf)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
