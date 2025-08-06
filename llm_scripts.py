from openai import OpenAI
from tool_scripts import CatalogEntrySuggestion, get_node_pipeline_source_code
from typing import List

client = OpenAI()


def format_context_for_llm(context_dict: dict[str, str]) -> str:
    return "\n\n".join(
        f"# {file}\n```python\n{content.strip()}\n```"
        for file, content in context_dict.items()
    )


def build_prompt(suggestions: List[CatalogEntrySuggestion], context_md: str | None) -> List[dict]:
    intro_lines = [
        "You are a Kedro expert. Based on the file paths and names below, suggest the most likely Kedro dataset type.",
        "Only suggest valid Kedro dataset class names (like `pandas.CSVDataset`, `pandas.ParquetDataset`, `json.JSONDataset`, `PickleDataset`, `ImageDataset`, `TextDataset`, etc).",
        "You MUST give a response for every entry — if you're unsure, make your best educated guess.",
        "Ignore entries that look like versioning metadata (e.g. `dataset_name/version.txt`, `.ipynb_checkpoints/`, `._SUCCESS`, `_versions`, etc). These are not datasets.",
        "",
        "Input format: name: filepath",
        "Output format: name: suggested_type",
        ""
    ]
    for s in suggestions:
        intro_lines.append(f"{s.suggested_name}: {s.filepath}")

    messages = [{"role": "system", "content": "\n".join(intro_lines)}]

    if context_md:
        messages.append({
            "role": "user",
            "content": "Here is some project source code context that may help you decide:\n\n" + context_md
        })

    return messages


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
