import multiprocessing as mp
import time
from queue import Empty

import draccus
import pandas as pd
from research.datasets_utils import HFSource, NotTrainMixDataConfig, load_huggingface_datasets
from transformers import AutoTokenizer


def _shard_dataset(ds, num_workers: int, worker_id: int):
    """
    Shard a huggingface dataset/iterable dataset for this worker.
    Falls back to a manual slice if shard is unsupported.
    """
    try:
        return ds.shard(num_shards=num_workers, index=worker_id, contiguous=True)
    except Exception:
        total = len(ds)
        start = worker_id * total // num_workers
        end = (worker_id + 1) * total // num_workers
        return ds.select(range(start, end))


def _worker(worker_id: int, num_workers: int, config: NotTrainMixDataConfig, queue: mp.Queue, stop_event: mp.Event):
    """Load a shard of each source, compute counts, and push results to the queue."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        for source in config.sources:
            ds = load_huggingface_datasets(source)
            ds = _shard_dataset(ds, num_workers=num_workers, worker_id=worker_id)

            for row in ds:
                if stop_event.is_set():
                    queue.put((worker_id, None))
                    return

                try:
                    count = len(tokenizer.apply_chat_template(row[config.chat_column]))
                except Exception as exc:  # noqa: BLE001
                    queue.put((worker_id, {"error": str(exc)}))
                    queue.put((worker_id, None))
                    return

                queue.put((worker_id, count))

        queue.put((worker_id, None))
    except Exception as exc:  # noqa: BLE001
        queue.put((worker_id, {"error": str(exc)}))
        queue.put((worker_id, None))


def _main_impl(config: NotTrainMixDataConfig, report_every: int = 10_000):
    counts: list[int] = []
    num_workers = config.num_workers or mp.cpu_count()

    print(f"Starting multiprocessing count with {num_workers} workers")
    queue = mp.Queue(maxsize=num_workers * 1000)
    stop_event = mp.Event()

    processes = []
    for i in range(num_workers):
        process = mp.Process(target=_worker, args=(i, num_workers, config, queue, stop_event))
        process.start()
        processes.append(process)

    start = time.time()
    last_report = start
    finished_workers = 0
    errors: list[str] = []

    try:
        while finished_workers < num_workers:
            try:
                worker_id, payload = queue.get(timeout=1.0)
            except Empty:
                continue

            if payload is None:
                finished_workers += 1
                print(f"Worker {worker_id} finished ({finished_workers}/{num_workers})")
                continue

            if isinstance(payload, dict) and "error" in payload:
                errors.append(f"Worker {worker_id}: {payload['error']}")
                stop_event.set()
                continue

            counts.append(payload)

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
    except KeyboardInterrupt:
        print("Interrupted by user, stopping workers...")
        stop_event.set()
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    if errors:
        print("Errors encountered during processing:")
        for err in errors:
            print(err)

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
    config = NotTrainMixDataConfig(
        sources=[
            HFSource(path="allenai/Dolci-Think-SFT", streaming=True),
        ],
        tokenizer_path="google/gemma-3-1b-it",
        chat_column="messages",
        num_workers=96,
    )
    main(config)
