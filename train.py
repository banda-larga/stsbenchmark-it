import math
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datasets import load_dataset
from pathlib import Path
from rich.logging import RichHandler
import logging
import click

import torch
from torch.utils.data import DataLoader

g = torch.Generator()
g.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


def get_input_example(row):
    """Sentence pair and score from STSbenchmark HuggingFace dataset"""
    score = float(row["similarity_score"]) / 5.0  # Normalize score to range 0 ... 1
    inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)
    return inp_example


def get_samples(dataset):
    """Get train, dev and test samples from the dataset
    Args:
        dataset: HuggingFace stsb dataset
    Returns:
        train_samples: list of InputExample
        dev_samples: list of InputExample
        test_samples: list of InputExample
    """

    train_samples = (
        dataset["train"].to_pandas().apply(get_input_example, axis=1).tolist()
    )
    dev_samples = dataset["dev"].to_pandas().apply(get_input_example, axis=1).tolist()
    test_samples = dataset["test"].to_pandas().apply(get_input_example, axis=1).tolist()

    return train_samples, dev_samples, test_samples


@click.command()
@click.option(
    "--model_name",
    default="bert-base-nli-mean-tokens",
    help="Name of the pre-trained model",
)
@click.option(
    "--dataset", default="stsb_multi_mt", help="HuggingFace STSbenchmark dataset"
)
@click.option("--train_batch_size", default=16, help="Batch size for training")
@click.option("--num_epochs", default=4, help="Number of epochs")
@click.option(
    "--output_path",
    default="output/training_stst",
    help="Path to save the trained model",
)
def main(model_name, dataset, train_batch_size, num_epochs, output_path):
    """
    Train a sentence transformer model on the STSbenchmark dataset
    """
    logging.info(f"Training model: {model_name}")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    word_embedding_model = models.Transformer(model_name)

    tokens = ["[DOC]", "[QRY]"]  # we later want to use these
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(
        len(word_embedding_model.tokenizer)
    )

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    log.info("Read STSbenchmark dataset")
    dataset = load_dataset(dataset, name="it")
    train_samples, dev_samples, test_samples = get_samples(dataset)

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size, generator=g
    )
    train_loss = losses.CosineSimilarityLoss(model=model)

    log.info("Read STSbenchmark dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, name="sts-dev"
    )

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    log.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
    )

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name="sts-test"
    )
    test_evaluator(model, output_path=output_path)

    model.save(output_path)


if __name__ == "__main__":
    main()
