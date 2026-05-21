import unittest

from src import datasets


class DatasetTargetEncodingTest(unittest.TestCase):
    def test_minority_class_is_positive_for_all_datasets(self):
        for dataset_number in range(1, 21):
            with self.subTest(dataset=f"df{dataset_number}"):
                loader = getattr(datasets, f"df{dataset_number}")
                df = loader()

                self.assertIn("target", df.columns)

                target_counts = df["target"].value_counts().to_dict()
                self.assertEqual(set(target_counts), {0, 1})
                self.assertLessEqual(target_counts[1], target_counts[0])


if __name__ == "__main__":
    unittest.main()
