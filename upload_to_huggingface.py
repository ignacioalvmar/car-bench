"""
Upload CAR-Bench dataset to Hugging Face Hub.

This script:
1. Converts Python Task objects → JSONL (split by train/test)
2. Copies mock data JSONL files into the repo structure
3. Converts large JSONL files to Parquet for better compression
4. Creates a README.md with YAML dataset config
5. Uploads everything to the HF dataset repo

Usage:
    python upload_to_huggingface.py
"""

import json
import os
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_REPO_ID = "johanneskirmayr/car-bench-dataset"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(
    PROJECT_ROOT, "docs", "reference_data", "tasks"
)
MOCK_DATA_DIR = os.path.join(
    PROJECT_ROOT, "docs", "reference_data", "mock_data"
)

# Files to convert to Parquet (>50 MB JSONL files benefit from compression)
# Note: pois.jsonl has a heterogeneous schema (varying nested keys) which
# makes Parquet conversion problematic. At 68 MB it's fine as JSONL.
CONVERT_TO_PARQUET = {
    "mock_data/navigation/routes_metadata.jsonl",
    "mock_data/navigation/routes_index.jsonl",
}

# ---------------------------------------------------------------------------
# Step 1: Convert Task Python objects → JSONL
# ---------------------------------------------------------------------------

def serialize_tasks_to_jsonl(output_dir: str):
    """Import Task lists from Python files, split by train/test, write JSONL."""
    import importlib
    import types

    # We need to be able to import from the project
    sys.path.insert(0, PROJECT_ROOT)

    # The car_voice_assistant __init__.py eagerly preloads ALL mock data (1.7M
    # routes), which is slow and unnecessary here.  We inject a dummy module to
    # short-circuit that import chain so only the lightweight types/context
    # modules are loaded.
    dummy = types.ModuleType("car_bench.envs.car_voice_assistant")
    dummy.__path__ = [
        os.path.join(PROJECT_ROOT, "car_bench", "envs", "car_voice_assistant")
    ]
    dummy.__package__ = "car_bench.envs.car_voice_assistant"
    sys.modules["car_bench.envs.car_voice_assistant"] = dummy

    # Also prevent mock_data from loading
    dummy_mock = types.ModuleType("car_bench.envs.car_voice_assistant.mock_data")
    dummy_mock.__path__ = [
        os.path.join(
            PROJECT_ROOT, "car_bench", "envs", "car_voice_assistant", "mock_data"
        )
    ]
    dummy_mock.__package__ = "car_bench.envs.car_voice_assistant.mock_data"
    sys.modules["car_bench.envs.car_voice_assistant.mock_data"] = dummy_mock

    # Import task files from docs/reference_data/tasks/ using file paths
    # (they still import car_bench types internally, which is fine since
    # sys.path includes PROJECT_ROOT)
    def _load_tasks_from_file(filepath, module_name):
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TASKS

    BASE_TASKS = _load_tasks_from_file(
        os.path.join(TASKS_DIR, "tasks_base.py"), "tasks_base"
    )
    DISAMBIGUATION_TASKS = _load_tasks_from_file(
        os.path.join(TASKS_DIR, "tasks_disambiguation.py"), "tasks_disambiguation"
    )
    HALLUCINATION_TASKS = _load_tasks_from_file(
        os.path.join(TASKS_DIR, "tasks_hallucination.py"), "tasks_hallucination"
    )

    # Load split definitions
    with open(os.path.join(TASKS_DIR, "task_splits.json"), "r") as f:
        splits = json.load(f)

    task_groups = {
        "base": {t.task_id: t for t in BASE_TASKS},
        "disambiguation": {t.task_id: t for t in DISAMBIGUATION_TASKS},
        "hallucination": {t.task_id: t for t in HALLUCINATION_TASKS},
    }

    tasks_out_dir = os.path.join(output_dir, "tasks")
    os.makedirs(tasks_out_dir, exist_ok=True)

    for group_name, tasks_by_id in task_groups.items():
        for split_suffix in ["train", "test"]:
            split_key = f"{group_name}_{split_suffix}"
            if split_key not in splits:
                print(f"  ⚠️  Split key '{split_key}' not found in task_splits.json, skipping")
                continue

            task_ids_in_split = splits[split_key]
            out_path = os.path.join(tasks_out_dir, f"{split_key}.jsonl")

            with open(out_path, "w", encoding="utf-8") as fout:
                for tid in task_ids_in_split:
                    if tid not in tasks_by_id:
                        print(f"  ⚠️  Task '{tid}' not found in {group_name} tasks")
                        continue
                    task = tasks_by_id[tid]
                    # Serialize using Pydantic's model_dump, then convert
                    # complex nested fields to JSON strings for cleaner HF schema
                    d = task.model_dump()
                    # Keep top-level simple fields as native types
                    # Serialize complex nested dicts/lists as JSON strings
                    d["context_init_config"] = json.dumps(
                        d["context_init_config"], ensure_ascii=False
                    )
                    d["actions"] = json.dumps(
                        d["actions"], ensure_ascii=False
                    )
                    if d.get("removed_part") is not None:
                        d["removed_part"] = json.dumps(
                            d["removed_part"], ensure_ascii=False
                        )
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")

            n_tasks = len(task_ids_in_split)
            print(f"  ✅ {out_path} ({n_tasks} tasks)")


# ---------------------------------------------------------------------------
# Step 2: Copy mock data files
# ---------------------------------------------------------------------------

def copy_mock_data(output_dir: str):
    """Copy JSONL mock data files into the output directory structure."""
    mock_out_dir = os.path.join(output_dir, "mock_data")

    # Navigation files
    nav_files = [
        "navigation/locations.jsonl",
        "navigation/pois.jsonl",
        "navigation/routes_index.jsonl",
        "navigation/routes_location_location.jsonl",
        "navigation/routes_location_poi.jsonl",
        "navigation/routes_poi_location.jsonl",
        "navigation/routes_metadata.jsonl",
        "navigation/weather.jsonl",
    ]
    # Productivity files
    prod_files = [
        "productivity_and_communication/calendars.jsonl",
        "productivity_and_communication/contacts.jsonl",
    ]

    for rel_path in nav_files + prod_files:
        src = os.path.join(MOCK_DATA_DIR, rel_path)
        dst = os.path.join(mock_out_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Use symlinks for large files to avoid slow copies
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  ✅ {rel_path} ({size_mb:.1f} MB) [symlinked]")


# ---------------------------------------------------------------------------
# Step 3: Convert large JSONL to Parquet
# ---------------------------------------------------------------------------

def convert_large_files_to_parquet(output_dir: str):
    """Convert specified large JSONL files to Parquet for better compression.

    Uses pyarrow directly instead of the datasets library because
    some JSONL files have heterogeneous schemas (e.g. routes_index has
    an optional 'on_id' column).  pyarrow unifies schemas gracefully.
    """
    import pyarrow.json as paj
    import pyarrow.parquet as pq

    for rel_path in CONVERT_TO_PARQUET:
        jsonl_path = os.path.join(output_dir, rel_path)
        if not os.path.exists(jsonl_path):
            print(f"  ⚠️  {rel_path} not found, skipping Parquet conversion")
            continue

        parquet_path = jsonl_path.replace(".jsonl", ".parquet")
        print(f"  Converting {rel_path} → Parquet...")

        # pyarrow.json.read_json handles heterogeneous schemas by unifying columns
        table = paj.read_json(jsonl_path)
        pq.write_table(table, parquet_path, compression="snappy")

        # Report size savings
        jsonl_size = os.path.getsize(jsonl_path) / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        print(
            f"  ✅ {rel_path}: {jsonl_size:.1f} MB → {parquet_size:.1f} MB "
            f"({(1 - parquet_size / jsonl_size) * 100:.0f}% reduction)"
        )

        # Remove the original JSONL to avoid uploading both
        os.remove(jsonl_path)


# ---------------------------------------------------------------------------
# Step 4: Create README.md with YAML config
# ---------------------------------------------------------------------------

README_CONTENT = """\
---
configs:
  # ── Tasks ──────────────────────────────────────────────────────────────
  - config_name: tasks_base
    default: true
    data_files:
      - split: train
        path: "tasks/base_train.jsonl"
      - split: test
        path: "tasks/base_test.jsonl"
  - config_name: tasks_disambiguation
    data_files:
      - split: train
        path: "tasks/disambiguation_train.jsonl"
      - split: test
        path: "tasks/disambiguation_test.jsonl"
  - config_name: tasks_hallucination
    data_files:
      - split: train
        path: "tasks/hallucination_train.jsonl"
      - split: test
        path: "tasks/hallucination_test.jsonl"

  # ── Mock Data: Navigation ──────────────────────────────────────────────
  - config_name: mock_locations
    data_files: "mock_data/navigation/locations.jsonl"
  - config_name: mock_pois
    data_files: "mock_data/navigation/pois.jsonl"
  - config_name: mock_weather
    data_files: "mock_data/navigation/weather.jsonl"
  - config_name: mock_routes_location_location
    data_files: "mock_data/navigation/routes_location_location.jsonl"
  - config_name: mock_routes_location_poi
    data_files: "mock_data/navigation/routes_location_poi.jsonl"
  - config_name: mock_routes_poi_location
    data_files: "mock_data/navigation/routes_poi_location.jsonl"
  - config_name: mock_routes_index
    data_files: "mock_data/navigation/routes_index.jsonl"
  - config_name: mock_routes_metadata
    data_files: "mock_data/navigation/routes_metadata.jsonl"

  # ── Mock Data: Productivity & Communication ────────────────────────────
  - config_name: mock_calendars
    data_files: "mock_data/productivity_and_communication/calendars.jsonl"
  - config_name: mock_contacts
    data_files: "mock_data/productivity_and_communication/contacts.jsonl"

license: mit
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - benchmark
  - car
  - voice-assistant
  - agentic
  - tool-use
  - function-calling
size_categories:
  - 1K<n<10K
---

# CAR-Bench Dataset

**CAR-Bench** is a benchmark for evaluating AI voice assistants in a realistic automotive (car) environment.
It tests an agent's ability to correctly use vehicle control tools, handle disambiguation, and avoid hallucinations.

## Dataset Structure

The dataset is organized into **task configs** and **mock data configs**:

### Tasks

Each task defines a user persona, an instruction, the initial vehicle/environment context, and the ground-truth sequence of tool-call actions the assistant should perform.

| Config | Description | Train | Test |
|--------|-------------|-------|------|
| `tasks_base` | Standard tasks covering vehicle controls, navigation, calendar, etc. | 50 | 50 |
| `tasks_disambiguation` | Tasks requiring the agent to disambiguate parameters (internally via preferences or by asking the user) | 30 | 26 |
| `tasks_hallucination` | Tasks where certain tools/parameters are intentionally removed to test if the agent hallucinates | 48 | 50 |

**Task schema:**

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique task identifier |
| `persona` | string | Description of the simulated user's personality and communication style |
| `calendar_id` | string | Reference to a calendar in the mock data |
| `instruction` | string | The instruction given to the simulated user |
| `context_init_config` | string (JSON) | Initial vehicle and environment state (battery, seats, location, weather, preferences, etc.) |
| `actions` | string (JSON) | Ground-truth sequence of tool calls `[{name, kwargs, index, dependent_on_action_index}]` |
| `task_type` | string | One of: `base`, `disambiguation_internal`, `disambiguation_user`, `hallucination_missing_tool`, `hallucination_missing_tool_parameter`, `hallucination_missing_tool_response` |
| `disambiguation_element_internal` | string or null | What needs to be disambiguated internally (only set in disambiguation tasks) |
| `disambiguation_element_user` | string or null | What needs to be disambiguated with the user (only set in disambiguation tasks) |
| `disambiguation_element_note` | string or null | Note explaining the disambiguation (only set in disambiguation tasks) |
| `removed_part` | string (JSON) or null | Which tools/parameters were removed (only set in hallucination tasks) |

### Mock Data

The mock data simulates a realistic car environment database used by the tools during benchmark execution.

| Config | Rows | Description |
|--------|------|-------------|
| `mock_locations` | 48 | European cities with GPS coordinates |
| `mock_pois` | 130,693 | Points of interest (airports, bakeries, restaurants, etc.) |
| `mock_weather` | 48 | Weather data per location (8 time-slots/day) |
| `mock_routes_location_location` | 6,768 | Routes between locations (3 alternatives each) |
| `mock_routes_location_poi` | 1,378 | Routes from locations to POIs |
| `mock_routes_poi_location` | 1,378 | Routes from POIs to locations |
| `mock_routes_index` | 1,763,870 | Route lookup index |
| `mock_routes_metadata` | 1,754,346 | Metadata for POI-to-POI route generation |
| `mock_calendars` | 100 | Calendar entries with meetings |
| `mock_contacts` | 100 | Contact information |

## Usage

```python
from datasets import load_dataset

# Load tasks
tasks = load_dataset("johanneskirmayr/car-bench-dataset", "tasks_base")
print(tasks["test"][0])

# Load mock data
locations = load_dataset("johanneskirmayr/car-bench-dataset", "mock_locations", split="train")
contacts = load_dataset("johanneskirmayr/car-bench-dataset", "mock_contacts", split="train")

# Parse nested JSON fields
import json
task = tasks["test"][0]
context = json.loads(task["context_init_config"])
actions = json.loads(task["actions"])
```

## Citation

If you use this dataset, please cite the CAR-Bench paper:

```bibtex
@misc{kirmayr2026carbenchevaluatingconsistencylimitawareness,
      title={CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty}, 
      author={Johannes Kirmayr and Lukas Stappen and Elisabeth André},
      year={2026},
      eprint={2601.22027},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.22027}, 
}
```
"""


def create_readme(output_dir: str):
    """Write the dataset card README.md."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_CONTENT)
    print(f"  ✅ README.md created")


# ---------------------------------------------------------------------------
# Step 5: Upload to Hugging Face Hub
# ---------------------------------------------------------------------------

def upload_to_hub(output_dir: str):
    """Upload the prepared directory to the HF dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"\n📤 Uploading to {HF_REPO_ID}...")
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print(f"\n🎉 Done! View at: https://huggingface.co/datasets/{HF_REPO_ID}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Create a temporary staging directory
    output_dir = os.path.join(PROJECT_ROOT, "_hf_upload_staging")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("=" * 60)
    print("CAR-Bench → Hugging Face Dataset Upload")
    print("=" * 60)

    print("\n📝 Step 1: Converting tasks (Python → JSONL)...")
    serialize_tasks_to_jsonl(output_dir)

    print("\n📁 Step 2: Copying mock data files...")
    copy_mock_data(output_dir)

    if "--parquet" in sys.argv:
        print("\n🔄 Step 3: Converting large JSONL → Parquet...")
        convert_large_files_to_parquet(output_dir)
    else:
        print("\n⏭️  Step 3: Skipping Parquet conversion (pass --parquet to enable)")

    print("\n📄 Step 4: Creating README.md with dataset config...")
    create_readme(output_dir)

    # Print staging directory summary
    print("\n📊 Staging directory contents:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"{subindent}{f}  ({size_mb:.1f} MB)")

    # Upload (use --yes flag or auto-confirm)
    print(f"\n🎯 Target repo: https://huggingface.co/datasets/{HF_REPO_ID}")
    if "--yes" in sys.argv:
        upload_to_hub(output_dir)
    else:
        try:
            confirm = input("Proceed with upload? [y/N]: ").strip().lower()
        except EOFError:
            confirm = "y"
        if confirm == "y":
            upload_to_hub(output_dir)
        else:
            print(f"\n⏸️  Upload skipped. Staged files are at: {output_dir}")
            print("   You can inspect them and re-run, or upload manually with:")
            print(f"   huggingface-cli upload {HF_REPO_ID} {output_dir} --repo-type dataset")


if __name__ == "__main__":
    main()
