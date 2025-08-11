from pathlib import Path
from openai import OpenAI
from models import CatalogEntrySuggestion
from typing import List

client = OpenAI()


def format_context_for_llm(context_dict: dict[str, str]) -> str:
    return "\n\n".join(
        f"# {file}\n```python\n{content.strip()}\n```"
        for file, content in context_dict.items()
    )


def get_node_pipeline_source_code(src_root: str = "src") -> dict[str, str]:
    relevant_files = {}
    for path in Path(src_root).rglob("*.py"):
        if path.name in {"nodes.py", "pipeline.py"}:
            try:
                relative_path = str(path.relative_to(src_root))
                relevant_files[relative_path] = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Skipping {path}: {e}")
    return relevant_files


def build_prompt(suggestions: List[CatalogEntrySuggestion], context_md: str | None) -> List[dict]:
    dataset_lines = [f"{s.suggested_name}: {s.filepath}" for s in suggestions]
    dataset_block = "\n".join(dataset_lines)

    instructions = [
        "You are a Kedro expert.",
        "Your task is to infer the most appropriate Kedro dataset type for each dataset listed below.",
        "",
        "Use the source code context to help guide your choices.",
        "Prioritize evidence from code (e.g., how a dataset is loaded, used, or saved).",
        "If no relevant code is found, fall back to the file extension and naming heuristics.",
        "",
        "Ignore versioning metadata or non-datasets (e.g. `.ipynb_checkpoints/`, `_versions/`, `dataset_name/version.txt`, etc).",
        "",
        "Only suggest **valid** Kedro dataset class names (e.g. `pandas.CSVDataset`, `pandas.ParquetDataset`, `json.JSONDataset`, `pickle.PickleDataset`, etc).",
        "",
        "You MUST respond with a dataset type for every entry — even if you're unsure, make your best guess.",
        "",
        "Dataset entries:",
        dataset_block,
    ]

    if context_md:
        instructions += [
            "",
            "Source code context:",
            context_md,
        ]

    instructions += [
        "",
        "Respond in this format only:",
        "dataset_name: dataset.DatasetType",
        "",
        "Do not explain your choices. Just output the name-to-type mapping.",
    ]

    return [
        {"role": "system", "content": "You are a helpful assistant that understands the Kedro data catalog and pipelines."},
        {"role": "user", "content": "\n".join(instructions)}
    ]


def parse_llm_response(content: str) -> dict[str, str | None]:
    type_map = {}
    for line in content.strip().splitlines():
        if ":" not in line:
            continue
        name, dataset_type = line.split(":", 1)
        type_map[name.strip()] = dataset_type.strip()
    return type_map


def is_noise_file(s: CatalogEntrySuggestion) -> bool:
    # Add filters for files that should NOT be treated as datasets
    noise_indicators = [
        "_versions", "version", "checkpoint", "SUCCESS", ".ipynb_checkpoints",
        ".DS_Store", "._", ".trash", "metadata", ".log"
    ]
    path = s.filepath.lower()
    return any(part in path for part in noise_indicators)


def infer_dataset_types(
    suggestions: List[CatalogEntrySuggestion],
    verbose: bool = True
) -> List[CatalogEntrySuggestion]:
    def log(msg: str):
        if verbose:
            print(msg)

    context_dict = get_node_pipeline_source_code()
    context_md = format_context_for_llm(context_dict)

    unresolved = suggestions.copy()
    final_results: dict[str, str] = {}

    # 🔍 Attempt with context immediately
    log("🔍 Attempting with source code context from the start...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=build_prompt(unresolved, context_md=context_md),
        temperature=0.2,
    )
    first_pass = parse_llm_response(response.choices[0].message.content)

    for s in unresolved:
        result = first_pass.get(s.suggested_name)
        if not result or result.lower() in {"unknown", "?", "none", "unsure", "null"}:
            log(f"  - ⚠️ LLM returned uncertain value for `{s.suggested_name}`, but keeping it anyway.")
        else:
            log(f"  - ✅ Resolved: {s.suggested_name} → {result}")
        final_results[s.suggested_name] = result

    # Apply final results
    for s in suggestions:
        s.suggested_type = final_results.get(s.suggested_name)

    log(f"\n🏁 Final dataset type mapping:")
    for s in suggestions:
        log(f"  - {s.suggested_name}: {s.suggested_type}")

    return suggestions
