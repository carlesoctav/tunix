#!/usr/bin/env python3
"""Check if datasets have the same columns."""

import sys
from typing import Optional
from datasets import load_dataset

# Dataset configurations based on your table
DATASETS = [
    {
        "name": "Everyday Conversations",
        "path": "everyday_conversations",  # Replace with actual HF path
        "subset": None,
        "split": "train",
    },
    {
        "name": "SystemChats 30k",
        "path": "systemchats_30k",  # Replace with actual HF path
        "subset": None,
        "split": "train",
    },
    {
        "name": "Tulu 3 SFT Personas IF",
        "path": "allenai/tulu-3-sft-personas-if",  # Replace with actual HF path
        "subset": None,
        "split": "train",
    },
    {
        "name": "Everyday Conversations (Qwen3-32B)",
        "path": "everyday_conversations_qwen3",  # Replace with actual HF path
        "subset": None,
        "split": "train",
    },
    {
        "name": "SystemChats 30k (Qwen3-32B)",
        "path": "systemchats_30k_qwen3",  # Replace with actual HF path
        "subset": None,
        "split": "train",
    },
    {
        "name": "s1k-1.1",
        "path": "s1k-1.1",  # Replace with actual HF path
        "subset": None,
        "split": "train",
    },
]


def check_dataset_columns():
    """Check if all datasets have the same columns."""
    all_columns: dict[str, Optional[list[str]]] = {}
    reference_columns: Optional[list[str]] = None
    reference_name: Optional[str] = None

    print("Checking dataset columns...\n")

    for dataset_config in DATASETS:
        name = dataset_config["name"]
        path = dataset_config["path"]
        subset = dataset_config["subset"]
        split = dataset_config["split"]

        try:
            print(f"Loading: {name}")
            if subset:
                ds = load_dataset(path, subset, split=split)
            else:
                ds = load_dataset(path, split=split)

            # Handle both Dataset and DatasetDict
            columns: list[str] = []
            if hasattr(ds, "column_names"):
                col_names = ds.column_names
                if isinstance(col_names, dict):
                    # DatasetDict case - get first split's columns
                    first_split = list(col_names.values())[0]
                    if first_split is not None:
                        columns = sorted(first_split)
                elif col_names is not None:
                    # Regular Dataset
                    columns = sorted(col_names)

            all_columns[name] = columns

            print(f"  Columns: {columns}")
            print(f"  Number of columns: {len(columns)}")
            try:
                num_rows = len(ds)  # type: ignore
                print(f"  Number of rows: {num_rows}\n")
            except:
                print(f"  Number of rows: unknown\n")

            # Set first dataset as reference
            if reference_columns is None:
                reference_columns = columns
                reference_name = name

        except Exception as e:
            print(f"  ❌ Error loading dataset: {e}\n")
            all_columns[name] = None

    # Compare all datasets to reference
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nReference dataset: {reference_name}")
    print(f"Reference columns: {reference_columns}\n")

    all_same = True
    for name, columns in all_columns.items():
        if columns is None:
            print(f"❌ {name}: Failed to load")
            all_same = False
            continue

        if columns == reference_columns:
            print(f"✅ {name}: Same columns")
        else:
            print(f"❌ {name}: Different columns")
            all_same = False

            # Show differences
            if reference_columns:
                missing = set(reference_columns) - set(columns)
                extra = set(columns) - set(reference_columns)

                if missing:
                    print(f"   Missing columns: {missing}")
                if extra:
                    print(f"   Extra columns: {extra}")

    print("\n" + "=" * 80)
    if all_same:
        print("✅ All datasets have the same columns!")
    else:
        print("❌ Datasets have different columns")
    print("=" * 80)

    return all_same


if __name__ == "__main__":
    success = check_dataset_columns()
    sys.exit(0 if success else 1)
