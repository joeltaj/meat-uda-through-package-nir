import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiments as E


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full, cls.holdout = E.load_saved_results(E.DEFAULT_DATA, E.EXAMPLE_RUN)

    def test_all_ten_tables_have_named_outputs(self):
        index = E.pd.read_csv(E.EXAMPLE_RUN / "table_index.csv")
        self.assertEqual(list(index.Table), list(range(1, 11)))
        for row in index.itertuples(index=False):
            table = E.pd.read_csv(E.EXAMPLE_RUN / row.CSV)
            self.assertGreater(len(table), 0)
            self.assertTrue(row.Title)

    def test_missing_domain_grid_and_nan_are_rejected(self):
        wrong = self.full.copy()
        wrong.loc[0, "mode"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "Incorrect domain"):
            E.validate_full_results(wrong)
        missing = self.full[self.full["split"] != 24]
        with self.assertRaisesRegex(ValueError, "All 25"):
            E.validate_full_results(missing)
        invalid = self.full.copy()
        invalid.loc[0, "f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Method failure"):
            E.validate_full_results(invalid)

    def test_a_different_dataset_cannot_reuse_results(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            provenance = json.loads((E.EXAMPLE_RUN / "provenance.json").read_text())
            provenance["data_sha256"] = "0" * 64
            (temporary / "provenance.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "different dataset"):
                E.load_saved_results(E.DEFAULT_DATA, temporary)

    def test_holdout_accuracy_and_macro_f1_from_actual_predictions(self):
        predictions = E.pd.read_csv(E.EXAMPLE_RUN / "holdout_predictions.csv")
        for mode, group in predictions.groupby("mode"):
            expected = self.holdout[self.holdout.Mode == mode].iloc[0]
            E.np.testing.assert_allclose(
                [100 * E.accuracy_score(group.truth, group.prediction),
                 100 * E.f1_score(group.truth, group.prediction, average="macro")],
                [expected.accuracy_pct, expected.scan_macro_f1_pct], rtol=0, atol=1e-12)

    def test_fresh_h2_split_matches_included_computed_scores(self):
        _, X, sid, mode, y, _ = E.R.load_data(E.DEFAULT_DATA)
        with E.threadpool_limits(limits=1):
            fresh = E.pd.DataFrame(E.R.evaluate_h2_split(0, X, sid, mode, y))
        saved = self.full[(self.full["split"] == 0) & self.full["mode"].isin(["TP", "TB"])
                          & self.full.method.isin(E.R.ABLATION_METHODS)]
        keys = ["mode", "frac", "method"]
        self.assertEqual(len(fresh), 98)
        E.np.testing.assert_allclose(fresh.sort_values(keys).f1,
                                   saved.sort_values(keys).f1, rtol=0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
