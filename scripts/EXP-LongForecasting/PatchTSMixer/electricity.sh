if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/PatchTSMixer" ]; then
    mkdir ./logs/LongForecasting/PatchTSMixer
fi
seq_len=336
model_name=PatchTSMixer
dataset=electricity
num_channels=321

for pred_len in 96 192 336 720
do
python3 -u run_longExp.py \
  --single_layer_mixer true \
  --num_blocks 1 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --patch_size 16 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len'.log'
done 