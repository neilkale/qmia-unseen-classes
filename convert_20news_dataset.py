import os
import time
import shutil
from typing import Optional, Tuple, List

from sklearn.datasets import fetch_20newsgroups


def get_local_cache_dir(base_dir: str = "./data") -> str:
    cache_dir = os.path.join(base_dir, "20newsgroups", "sklearn_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_rank() -> int:
    rank_env = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank_env)
    except Exception:
        return 0


def _fetch_with_retry(subset: str, remove: tuple, categories: Optional[List[str]], data_home: str):
    try:
        return fetch_20newsgroups(subset=subset, remove=remove, categories=categories, data_home=data_home)
    except FileNotFoundError:
        shutil.rmtree(data_home, ignore_errors=True)
        os.makedirs(data_home, exist_ok=True)
        return fetch_20newsgroups(subset=subset, remove=remove, categories=categories, data_home=data_home)


def prepare_20news_split(
    subset: str,
    remove: tuple = ("headers", "footers", "quotes"),
    categories: Optional[List[str]] = None,
    base_dir: str = "./data",
) -> Tuple[list, list]:
    """
    Prepare the 20 Newsgroups texts and labels into the local cache in a DDP-safe way.
    Returns (texts, labels).
    """
    data_home = get_local_cache_dir(base_dir)
    target_dir = os.path.join(data_home, "20news_home")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    rank = _get_rank()
    if rank == 0:
        ds = _fetch_with_retry(subset=subset, remove=remove, categories=categories, data_home=data_home)
    else:
        timeout_s = 300
        start_t = time.time()
        while not os.path.isdir(target_dir) and (time.time() - start_t) < timeout_s:
            time.sleep(1.0)
        if not os.path.isdir(target_dir):
            ds = _fetch_with_retry(subset=subset, remove=remove, categories=categories, data_home=data_home)
        else:
            ds = fetch_20newsgroups(subset=subset, remove=remove, categories=categories, data_home=data_home)

    texts = [t.replace("\\", " ") for t in ds.data]
    labels = list(ds.target)
    return texts, labels


def main():
    base_dir = os.environ.get("DATA_DIR", "./data")
    out_dir = os.path.join(base_dir, "20newsgroups", "prepared")

    def write_per_file(split: str, texts: List[str], labels: List[int]):
        split_dir = os.path.join(out_dir, split)
        # Create class directories class_1 ... class_20
        for y in set(labels):
            class_dir = os.path.join(split_dir, f"class_{int(y)+1}")
            os.makedirs(class_dir, exist_ok=True)
        # Write each example into its class directory
        counters = {}
        for t, y in zip(texts, labels):
            cls = int(y) + 1
            counters[cls] = counters.get(cls, 0) + 1
            fname = f"{counters[cls]:08d}.txt"
            fpath = os.path.join(split_dir, f"class_{cls}", fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(t)

    # Train
    train_texts, train_labels = prepare_20news_split(subset="train", base_dir=base_dir)
    write_per_file("train", train_texts, train_labels)

    # Test
    test_texts, test_labels = prepare_20news_split(subset="test", base_dir=base_dir)
    write_per_file("test", test_texts, test_labels)

    print(f"Wrote prepared 20 Newsgroups (per-file) to {out_dir}")


if __name__ == "__main__":
    main()


