import os
from pathlib import Path
import re
from llm_scripts import infer_dataset_types
from models import ScannedDataFile, ObservedProject, CatalogEntrySuggestion
from typing import List

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


def scan_data_folder(data_dir: str = "data") -> List[ScannedDataFile]:
    entries = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, data_dir)

            ext = Path(file).suffix.lower()
            dataset_type = EXT_TO_KEDRO_DATASET.get(ext)

            entries.append(ScannedDataFile(
                full_path=full_path,
                rel_path=rel_path,
                dataset_type=dataset_type 
            ))

    return entries


def observe_project(scanned_files: List[ScannedDataFile]) -> ObservedProject:
    versioned_files = []
    uncatalogued_files = []
    possible_models = []

    catalogued_files = set()
    versioned_pattern = re.compile(r"(.*)/\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2}\.\d{3}Z/.*")

    for entry in scanned_files:
        rel_path = entry.rel_path
        full_path = entry.full_path
        dataset_type = entry.dataset_type

        # Track versioned files
        if versioned_pattern.match(rel_path):
            versioned_files.append(rel_path)
            if "model" in rel_path.lower() or "regressor" in rel_path.lower():
                possible_models.append(rel_path)
            continue

        # Track uncatalogued files
        uncatalogued_files.append(rel_path)

    return ObservedProject(
        versioned_files=sorted(versioned_files),
        uncatalogued_files=sorted(uncatalogued_files),
        possible_models=sorted(possible_models),
    )


def analyze_observed_project(project: ObservedProject) -> List[CatalogEntrySuggestion]:
    suggestions = []

    versioned_dirs = {
        Path(f).parts[0:2] 
        for f in project.versioned_files
    }
    unique_versioned_paths = {os.path.join(*parts) for parts in versioned_dirs}

    # Combine unversioned + deduplicated versioned
    all_relevant_paths = project.uncatalogued_files + list(unique_versioned_paths)

    for rel_path in all_relevant_paths:
        _, ext = os.path.splitext(rel_path)
        #dataset_type = EXT_TO_KEDRO_DATASET.get(ext.lower(), None)
        dataset_type = None
        is_versioned = rel_path in unique_versioned_paths

        suggested_name = os.path.splitext(os.path.basename(rel_path))[0]

        suggestions.append(
            CatalogEntrySuggestion(
                filepath=rel_path,
                suggested_name=suggested_name,
                suggested_type=dataset_type,
                is_versioned=is_versioned,
            )
        )

    return suggestions


def to_catalog_entries(suggestions: List[CatalogEntrySuggestion]) -> dict:
    catalog = {}

    for entry in suggestions:
        dataset_name = entry.suggested_name.lower()
        dataset_type = entry.suggested_type
        rel_path = entry.filepath.replace("\\", "/")

        if dataset_type is None:
            print(f"[SKIPPED] Could not determine dataset type for: {rel_path}")
            continue

        catalog[dataset_name] = {
            "type": dataset_type,
            "filepath": f"data/{rel_path}"
        }

        if entry.is_versioned:
            catalog[dataset_name]["versioned"] = True

    return catalog


def write_catalog_to_yaml(catalog_dict: dict, output_path: str = "conf/base/auto_catalog.yml"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for i, (name, entry) in enumerate(catalog_dict.items()):
            yaml.dump({name: entry}, f, sort_keys=False)
            if i < len(catalog_dict) - 1:
                f.write("\n")


def update_auto_catalog():
    scanned_datafile: List[ScannedDataFile] = scan_data_folder()
    observed_project: ObservedProject = observe_project(scanned_datafile)
    catalog_plan: List[CatalogEntrySuggestion] = analyze_observed_project(observed_project)
    suggestions = infer_dataset_types(catalog_plan)
    catalog_entries = to_catalog_entries(suggestions)
    write_catalog_to_yaml(catalog_entries)