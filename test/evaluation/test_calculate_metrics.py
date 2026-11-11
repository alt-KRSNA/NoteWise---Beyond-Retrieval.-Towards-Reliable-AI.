
import unittest

from test.evaluation.calculate_metrics import compute_metrics


class TestCalculateMetrics(unittest.TestCase):

    def setUp(self):
        self.annotations = {
            "1": {
                "first_supporting_rank": 1,
                "label": "direct_support",
            },
            "2": {
                "first_supporting_rank": 2,
                "label": "direct_support",
            },
            "3": {
                "first_supporting_rank": None,
                "label": "no_support",
            },
        }

        self.results = {
            "results": [
                {"case_id": 1},
                {"case_id": 2},
                {"case_id": 3},
            ]
        }

    def test_top1_support_rate(self):
        metrics = compute_metrics(self.annotations, self.results)

        self.assertAlmostEqual(
            metrics["top1_support_rate"], 1 / 3
        )

    def test_top3_support_rate(self):
        metrics = compute_metrics(self.annotations, self.results)

        self.assertAlmostEqual(
            metrics["top3_support_rate"], 2 / 3
        )

    def test_mrr_at_3(self):
        metrics = compute_metrics(self.annotations, self.results)

        self.assertAlmostEqual(
            metrics["mrr_at_3"], 0.5
        )

    def test_empty_dataset(self):
        with self.assertRaises(ValueError):
            compute_metrics(
                self.annotations,
                {"results": []}
            )

    def test_evidence_category_counts(self):
        metrics = compute_metrics(self.annotations, self.results)

        self.assertEqual(
            metrics["label_counts"],
            {
                "direct_support": 2,
                "related_only": 0,
                "no_support": 1,
            }
        )


if __name__ == "__main__":
    unittest.main()
