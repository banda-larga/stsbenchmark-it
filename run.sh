#!/bin/bash

if [ -f "requirements.txt" ]; then
    echo "Installing requirements"
    pip3 install -r requirements.txt
fi

models=(
    "dbmdz/electra-base-italian-mc4-cased-discriminator"
    "dbmdz/electra-base-italian-xxl-cased-discriminator"
    "indigo-ai/BERTino"
    "dbmdz/bert-base-italian-xxl-cased"
    "dbmdz/bert-base-italian-xxl-uncased"
    "Musixmatch/umberto-commoncrawl-cased-v1"
    "Musixmatch/umberto-wikipedia-uncased-v1"
)


echo "Banchmarking Italian models on STS-B"

for model in "${models[@]}"
do
    echo "Training model $model"
    python3 train.py --model_name "$model" --num_epochs 4 --output_path "output/$(echo $model | awk -F '/' '{print $NF}')"
done

python3 get_results.py --path output

echo "Done!"