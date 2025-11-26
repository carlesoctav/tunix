import multiprocessing as mp

import draccus
import pandas as pd
from research.datasets_utils import HFSource, TrainMixDataConfig
from tqdm import tqdm



def _main_impl(config: TrainMixDataConfig):
    ds = config.make()
    count = []
    for input in tqdm(ds):
        count.append(ds)

    series = pd.Series(count)
    desc = series.describe()
    print(desc)
    return desc


@draccus.wrap()
def main(config: TrainMixDataConfig):
    return _main_impl(config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = TrainMixDataConfig(
        sources=[
            HFSource(path="allenai/Dolci-Think-SFT", streaming=True),
        ],
        tokenizer_path="google/gemma-3-1b-it",
        chat_column="messages",
        num_workers=96,
        per_worker_buffer_size = 1000,
    )
    main(cfg)
