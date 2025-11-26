import multiprocessing as mp
import time

import draccus
import pandas as pd
from research.datasets_utils import HFSource, NotTrainMixDataConfig
from research.mp import MPOptions, MultiProcessingIterDataset
from tqdm import tqdm


def _main_impl(config: NotTrainMixDataConfig, report_every: int = 10_000):
    # Build the dataset once; workers will inherit via fork.
    base_ds = config.make()

    # Wrap with our multiprocessing iterator.
    mp_ds = MultiProcessingIterDataset(
        base_ds,
        worker_init_fn=None,
        options=MPOptions(
            num_workers=config.num_workers or mp.cpu_count(),
            per_worker_buffer_size=getattr(config, "per_worker_buffer_size", 1000),
            batch_flush_size=1000,
            contiguous_shards=True,
        ),
    )

    counts: list[int] = []
    start = time.time()
    last_report = start
    for item in tqdm(mp_ds, desc="Counting tokens"):
        count = item["count"] if isinstance(item, dict) and "count" in item else item
        counts.append(count)
        if report_every and len(counts) % report_every == 0:
            now = time.time()
            elapsed = now - start
            window_rate = report_every / (now - last_report) if now > last_report else 0.0
            avg_rate = len(counts) / elapsed if elapsed else 0.0
            print(
                f"Progress: {len(counts):,} rows | "
                f"avg {avg_rate:,.2f} rows/s | "
                f"recent {window_rate:,.2f} rows/s | "
                f"elapsed {elapsed:,.1f}s"
            )
            last_report = now

    if not counts:
        print("No counts collected.")
        return None

    series = pd.Series(counts)
    desc = series.describe()
    print(desc)
    return desc


@draccus.wrap()
def main(config: NotTrainMixDataConfig, report_every: int = 10_000):
    return _main_impl(config, report_every=report_every)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = NotTrainMixDataConfig(
        sources=[
            HFSource(path="allenai/Dolci-Think-SFT", streaming=True),
        ],
        tokenizer_path="google/gemma-3-1b-it",
        chat_column="messages",
        num_workers=96,
        per_worker_buffer_size=1000,
    )
    main(cfg)
