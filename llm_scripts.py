from openai import OpenAI
from tool_scripts import CatalogEntrySuggestion
from typing import List

client = OpenAI()


def infer_dataset_types(suggestions: List[CatalogEntrySuggestion]) -> List[CatalogEntrySuggestion]:
    entries_to_infer = [s for s in suggestions if s.suggested_type is None]

    if not entries_to_infer:
        return suggestions  # Nothing to infer

    prompt_lines = [
        "You are a Kedro expert. Based on the file paths and names below, suggest the most likely Kedro dataset type.",
        "Only suggest valid Kedro dataset class names (like `pandas.CSVDataset`, `pandas.ExcelDataset`, `json.JSONDataset`, `PickleDataset`, `ImageDataset`, etc).",
        "If you can’t determine with confidence, reply with `null`.",
        "",
        "Input format: name: filepath",
        "Example output format: name: suggested_type",
        ""
    ]

    for s in entries_to_infer:
        prompt_lines.append(f"{s.suggested_name}: {s.filepath}")

    system_prompt = "\n".join(prompt_lines)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.2
    )

    reply = response.choices[0].message.content.strip()

    type_map = {}
    for line in reply.splitlines():
        if not line.strip() or ":" not in line:
            continue
        name, dataset_type = line.split(":", 1)
        name = name.strip()
        dataset_type = dataset_type.strip()
        type_map[name] = dataset_type if dataset_type != "null" else None

    for s in suggestions:
        if s.suggested_name in type_map and s.suggested_type is None:
            s.suggested_type = type_map[s.suggested_name]

    return suggestions