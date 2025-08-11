from pydantic import BaseModel
from typing import List


class DataFile(BaseModel):
    full_path: str
    rel_path: str
    ext: str


class ObservedProject(BaseModel):
    versioned_files: List[str]
    uncatalogued_files: List[str]
    possible_models: List[str]


class ScannedDataFile(BaseModel):
    full_path: str
    rel_path: str
    dataset_type: str | None


class CatalogEntrySuggestion(BaseModel):
    filepath: str
    suggested_name: str
    suggested_type: str | None
    is_versioned: bool