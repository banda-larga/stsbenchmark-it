echo "Banchmarking Italian models on STS--B"

model_name=dbmdz/electra-base-italian-mc4-cased-discriminator
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/electra-mc4

model_name=dbmdz/electra-base-italian-xxl-cased-discriminator
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/electra-xxl

model_name=indigo-ai/BERTino
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/bertino

model_name=dbmdz/bert-base-italian-xxl-cased
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/bert-xxl-cased

model_name=dbmdz/bert-base-italian-xxl-uncased
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/bert-xxl-uncased

model_name=Musixmatch/umberto-commoncrawl-cased-v1
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/umberto-commoncrawl

model_name=Musixmatch/umberto-wikipedia-uncased-v1
echo "Training model $model_name"
python3 train.py --model_name $model_name --num_epochs 4 --output_path output/umberto-wikipedia

python3 get_results.py --path output

echo "Done!"