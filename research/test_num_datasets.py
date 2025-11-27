import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ["HF_HOME"] = "/mnt/carles/.cache"

from research.datasets_utils import TrainMixDataConfig, HFSource


def main():
    config = TrainMixDataConfig(
        sources=[
            HFSource(
                path="allenai/Dolci-Think-SFT",
                streaming=False,
            )
        ],
        tokenizer_path="google/gemma-3-1b-it",
        batch_size=None,  # Yield individual items
        num_workers=96,
        per_worker_buffer_size=1000,
        use_fast_mp=True,  # Enable fast multiprocessing
        chat_column="messages",
        max_length=2048,
    )

    print("Initializing dataset...")
    ds = config.make()

    start = time.time()
    last = start
    total = 0

    # We will track both raw length (to check padding) and effective length (non-zero/non-pad)
    # Assuming 0 is pad for Gemma/sentencepiece, or at least a good proxy for "empty".
    raw_lengths = []
    effective_lengths = []

    report_every = 10000

    try:
        print("Starting iteration (this might take a moment to spin up workers)...")
        for item in tqdm(ds):
            input_ids = item["input_ids"]

            r_len = len(input_ids)
            # Count non-zero tokens as effective length (assuming 0 is pad)
            e_len = np.count_nonzero(input_ids)

            raw_lengths.append(r_len)
            effective_lengths.append(e_len)

            total += 1

            if report_every and total % report_every == 0:
                now = time.time()
                elapsed = now - start
                # Simple stats for the progress bar log
                # (tqdm handles the visual bar, but we add explicit log if needed,
                # though the user's snippet had manual printing, tqdm is cleaner)
                pass

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        elapsed = time.time() - start
        print(
            f"\nFinished processing {total:,} rows in {elapsed:,.1f}s ({total/elapsed:,.2f} rows/s avg)"
        )

        if total > 0:
            print("\n--- Raw Lengths (should be constant if padded) ---")
            print(pd.Series(raw_lengths).describe())

            print("\n--- Effective Lengths (non-zero tokens) ---")
            series = pd.Series(effective_lengths)
            print(series.describe())

            # Optional: Show some percentiles
            print("\nPercentiles (Effective Length):")
            print(series.quantile([0.5, 0.9, 0.95, 0.99, 1.0]))


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' to match train.py fixes
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    main()
