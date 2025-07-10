if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/PatchConv" ]; then
    mkdir ./logs/LongForecasting/PatchConv
fi
model_name=PatchConv
seq_len=336
dataset=ETTm2
num_channels=7
patch_size=16

for pred_len in 96 192 336 720
do
python3 -u run_longExp.py \
  --use_gpu false \
  --single_layer_mixer true \
  --stride 1 \
  --kernel_size 1 \
  --num_blocks 1 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --patch_size $patch_size \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len'.log'
done