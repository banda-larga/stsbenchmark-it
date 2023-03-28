import click
from pathlib import Path
import pandas as pd
from rich import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler()],
)


@click.command()
@click.option(
    "--path",
    default="output/",
    help="Path where the models are stored",
)
def main(path):
    scores = pd.DataFrame(columns=["model", "score"])
    models = [x for x in Path(path).iterdir() if x.is_dir()]
    for model in models:
        result_file = model / "eval" / "similarity_evaluation_sts-dev_results.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            last_row = df.iloc[-1].drop(["epoch", "steps"])
            avg = last_row.mean()
            scores = pd.concat(
                [scores, pd.DataFrame({"model": model.name, "score": avg}, index=[0])]
            )
    scores = scores.sort_values(by="score", ascending=False)
    scores.to_csv("results/scores.csv", index=False)

    logging.info(f"Saved!")


if __name__ == "__main__":
    main()
