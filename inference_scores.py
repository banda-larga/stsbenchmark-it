from sentence_transformers import SentenceTransformer
import logging
import pandas as pd
from pathlib import Path
from rich.logging import RichHandler
import click
from datasets import load_dataset
import time
import gc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


@click.command()
@click.option(
    "--path",
    default="output/",
    help="Path where the models are stored",
)
def main(path):
    scores = pd.DataFrame(columns=["model", "samples/sec"])
    models = [x for x in Path(path).iterdir() if x.is_dir()]

    dataset = load_dataset("stsb_multi_mt", name="it")
    train_samples = dataset["train"].to_pandas()
    samples = train_samples[["sentence1", "sentence2"]].values.flatten().tolist()
    log.info(f"Loaded {len(samples)} samples")

    for model in models:
        score = get_score(model, samples)
        scores = pd.concat(
            [
                scores,
                pd.DataFrame({"model": model.name, "samples/sec": score}, index=[0]),
            ]
        )

    scores = scores.sort_values(by="samples/sec", ascending=False)
    scores.to_csv("inference_scores.csv", index=False)
    log.info(f"Saved!")


def get_score(model, samples):
    model = SentenceTransformer(model)
    model.to("cpu")

    log.info("Inference time on training set")

    deltas = []
    for _ in range(10):
        start = time.time()
        model.encode(samples, batch_size=32, show_progress_bar=False)
        end = time.time()
        deltas.append(end - start)

    delta = sum(deltas) / len(deltas)
    samples_per_sec = len(samples) / delta

    log.info(f"Samples/sec: {samples_per_sec}")
    del model
    gc.collect()
    return samples_per_sec


if __name__ == "__main__":
    main()
