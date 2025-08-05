import os
from pathlib import Path
from llm_scripts import infer_kedro_dataset_type

import yaml


EXT_TO_KEDRO_DATASET = {
    ".csv": "pandas.CSVDataset",
    ".parquet": "pandas.ParquetDataset",
    ".xlsx": "pandas.ExcelDataset",
    ".xls": "pandas.ExcelDataset",
    ".xml": "pandas.XMLDataset",
    ".yaml": "yaml.YAMLDataset",
    ".yml": "yaml.YAMLDataset",
}


def scan_data_folder(data_dir: str = "data", use_llm: bool = True):
    entries = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, data_dir)
            dataset_type = EXT_TO_KEDRO_DATASET.get(ext)

            if use_llm and not dataset_type:
                try:
                    sample_lines = open(full_path).readlines()[:5]
                    file_sample = "".join(sample_lines)
                    dataset_type = infer_kedro_dataset_type(file, file_sample)
                except Exception as e:
                    print(f"[LLM Fallback Failed] {file}: {e}")
                    dataset_type = None

            if dataset_type:
                entries.append({
                    "path": full_path,
                    "rel_path": rel_path,
                    "dataset_type": dataset_type
                })

    return entries


def to_catalog_entries(file_info: list[dict]) -> dict:
    catalog = {}

    for entry in file_info:
        filename = Path(entry["rel_path"]).stem.lower()
        dataset_type = entry["dataset_type"]
        rel_path = entry["rel_path"].replace("\\", "/")  # Windows-safe

        if dataset_type is None:
            print(f"[SKIPPED] Could not determine dataset type for: {rel_path}")
            continue

        catalog[filename] = {
            "type": dataset_type,
            "filepath": f"data/{rel_path}"
        }

    return catalog


def write_catalog_to_yaml(catalog_dict: dict, output_path: str = "conf/base/auto_catalog.yml"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for i, (name, entry) in enumerate(catalog_dict.items()):
            yaml.dump({name: entry}, f, sort_keys=False)
            if i < len(catalog_dict) - 1:
                f.write("\n")


file_info = scan_data_folder()
catalog_dict = to_catalog_entries(file_info)
write_catalog_to_yaml(catalog_dict)