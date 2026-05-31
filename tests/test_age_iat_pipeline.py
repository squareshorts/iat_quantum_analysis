from __future__ import annotations

import unittest

import pandas as pd

from src.data.download_age_iat import select_best_candidates
from src.data.osf_utils import OSF_API_ROOT, classify_file_level, discover_osf_files_recursive
from src.data.prepare_age_iat import filter_critical_blocks, standardize_age_iat_columns


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, url, timeout=60, stream=False):
        return FakeResponse(self.mapping[url])


class AgeIatPipelineTests(unittest.TestCase):
    def test_discover_osf_manifest_from_mocked_api(self) -> None:
        folder_url = f"{OSF_API_ROOT}/files/folder-1/children/"
        mapping = {
            f"{OSF_API_ROOT}/nodes/cv7iq/": {
                "data": {"attributes": {"title": "Age IAT"}},
            },
            f"{OSF_API_ROOT}/nodes/cv7iq/files/osfstorage/": {
                "data": [],
                "links": {"next": None},
            },
            f"{OSF_API_ROOT}/nodes/cv7iq/children/": {
                "data": [
                    {
                        "id": "9jvmk",
                        "attributes": {"title": "Raw Data"},
                    }
                ],
                "links": {"next": None},
            },
            f"{OSF_API_ROOT}/nodes/9jvmk/files/osfstorage/": {
                "data": [
                    {
                        "attributes": {
                            "name": "years",
                            "materialized_path": "/years/",
                            "size": None,
                            "kind": "folder",
                        },
                        "relationships": {
                            "files": {
                                "links": {
                                    "related": {
                                        "href": folder_url,
                                    }
                                }
                            }
                        },
                        "links": {"download": None, "info": "", "html": ""},
                    }
                ],
                "links": {"next": None},
            },
            folder_url: {
                "data": [
                    {
                        "attributes": {
                            "name": "Age_iat_2019.zip",
                            "materialized_path": "/years/Age_iat_2019.zip",
                            "size": 123,
                            "kind": "file",
                        },
                        "relationships": {},
                        "links": {"download": "https://osf.io/download/example/", "info": "", "html": ""},
                    }
                ],
                "links": {"next": None},
            },
            f"{OSF_API_ROOT}/nodes/9jvmk/children/": {
                "data": [],
                "links": {"next": None},
            },
        }

        manifest = discover_osf_files_recursive("cv7iq", session=FakeSession(mapping))
        self.assertEqual(len(manifest), 2)
        zip_row = manifest.loc[manifest["name"].eq("Age_iat_2019.zip")].iloc[0]
        self.assertEqual(zip_row["source_node_title"], "Raw Data")
        self.assertEqual(zip_row["looks_like_level"], "trial")

    def test_trial_vs_summary_classification(self) -> None:
        trial_level, _ = classify_file_level(
            name="Age_iat_2019.zip",
            node_title="Raw Data",
            columns=["session_id", "block_number", "trial_number", "trial_latency"],
        )
        summary_level, _ = classify_file_level(
            name="Age IAT.public.2019.csv",
            node_title="Datasets & Codebooks",
            columns=["session_id", "D_biep.White_Good_all", "Mn_RT_all_3467"],
        )
        self.assertEqual(trial_level, "trial")
        self.assertEqual(summary_level, "summary")

    def test_standardize_age_like_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "session_id": ["a1", "a1"],
                "block_number": ["3", "4"],
                "trial_number": ["1", "2"],
                "trial_latency": ["450", "510"],
                "trial_error": ["0", "1"],
                "trial_name": ["Happy", "Old"],
                "block_pairing_definition": ["Young/Bad,Old/Good", "Young/Bad,Old/Good"],
            }
        )
        standardized, mapping = standardize_age_iat_columns(raw)
        self.assertEqual(mapping["pid"], "session_id")
        self.assertEqual(mapping["block"], "block_number")
        self.assertEqual(mapping["trial_in_block"], "trial_number")
        self.assertEqual(mapping["rt"], "trial_latency")
        self.assertIn("stimulus", standardized.columns)
        self.assertIn("category", standardized.columns)
        self.assertIn("trial_error", standardized.columns)

    def test_filter_critical_blocks(self) -> None:
        df = pd.DataFrame(
            {
                "pid": ["p1", "p1", "p1", "p1"],
                "block": [1, 3, 6, 7],
                "trial_in_block": [1, 2, 3, 4],
                "rt": [500, 600, 700, 800],
            }
        )
        filtered = filter_critical_blocks(df, [3, 4, 6, 7])
        self.assertEqual(filtered["block"].tolist(), [3, 6, 7])

    def test_select_best_candidate_fails_without_trial_level(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "name": "Age IAT.public.2019.csv.zip",
                    "path": "/Age IAT.public.2019.csv.zip",
                    "size": 100,
                    "kind": "file",
                    "download_url": "https://osf.io/download/summary/",
                    "source_node_title": "Datasets & Codebooks",
                    "looks_like_level": "summary",
                    "inferred_file_type": "csv.zip",
                }
            ]
        )
        with self.assertRaises(RuntimeError):
            select_best_candidates(manifest, year="2019")


if __name__ == "__main__":
    unittest.main()
