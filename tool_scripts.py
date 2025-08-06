import os
from pathlib import Path
from llm_scripts import infer_kedro_dataset_type
from pydantic import BaseModel
from typing import List, Set

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


TEXT_BASED_EXTENSIONS = {".csv", ".json", ".txt", ".yaml", ".yml", ".xml", ".md", ".log", ".py"}


class DataFile(BaseModel):
    full_path: str
    rel_path: str
    ext: str


class ObservedProject(BaseModel):
    data_files: List[DataFile]


class AnalysisResult(BaseModel):
    versioned_files: List[str]
    uncatalogued_files: List[str]
    possible_models: List[str]


def scan_data_folder(data_dir: str = "data", use_llm: bool = True):
    entries = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, data_dir)

            ext = Path(file).suffix.lower()
            dataset_type = EXT_TO_KEDRO_DATASET.get(ext)

            if use_llm and not dataset_type:
                try:
                    if ext in TEXT_BASED_EXTENSIONS:
                        with open(full_path, encoding="utf-8") as f:
                            sample_lines = f.readlines()[:5]
                        file_sample = "".join(sample_lines)
                    else:
                        file_sample = f"[binary file: {ext}]"

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


def update_auto_catalog(
    data_dir: str = "data",
    catalog_output: str = "conf/base/auto_catalog.yml",
    use_llm: bool = True,
) -> str:
    """
    Scans the data directory, infers dataset types, and updates the Kedro catalog.

    Args:
        data_dir: Path to the data directory.
        catalog_output: Output YAML path for the catalog entries.
        use_llm: Whether to use LLM for type inference if extension is unknown.

    Returns:
        The path to the generated catalog file.
    """
    file_info = scan_data_folder(data_dir=data_dir, use_llm=use_llm)
    catalog_dict = to_catalog_entries(file_info)
    write_catalog_to_yaml(catalog_dict, output_path=catalog_output)
    return catalog_output


def observe_project(data_dir: str = "data") -> dict:
    files = []

    for root, _, file_names in os.walk(data_dir):
        for file in file_names:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, data_dir)
            ext = Path(file).suffix.lower()

            files.append({
                "full_path": full_path,
                "rel_path": rel_path,
                "ext": ext,
            })

    return ObservedProject(data_files=files)


def analyze_observed_project(project: ObservedProject) -> AnalysisResult:
    versioned = []
    uncatalogued = []
    models = []

    for f in project.data_files:
        is_versioned = (
            f.rel_path.count("/") >= 2 and 
            f.rel_path.split("/")[-2].startswith("20")
        )
        is_model = f.ext in {".pickle", ".joblib", ".pkl", ".pt", ".pth"}

        if is_versioned:
            versioned.append(f.rel_path)

        if is_model:
            models.append(f.rel_path)

        if not is_versioned and not is_model:
            uncatalogued.append(f.rel_path)

    return AnalysisResult(
        versioned_files=versioned,
        uncatalogued_files=uncatalogued,
        possible_models=models,
    )
