import click
from pathlib import Path
import pandas as pd
from rich.logging import RichHandler
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
def main(path: str) -> None:
    """Run similarity evaluation for all models in the given directory."""
    scores = pd.DataFrame(columns=["model", "avg"])
    models = [x for x in Path(path).iterdir() if x.is_dir()]
    for model in models:
        result_file: Path = model / "eval" / "similarity_evaluation_sts-dev_results.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            last_row = df.iloc[-1].drop(["epoch", "steps"]).to_dict()
            last_row["avg"] = sum(last_row.values()) / len(last_row)
            scores = pd.concat(
                [scores, pd.DataFrame({"model": model.name, **last_row}, index=[0])]
            )
    scores = scores.sort_values(by="avg", ascending=False)
    scores.to_csv("scores.csv", index=False)


if __name__ == "__main__":
    main()
