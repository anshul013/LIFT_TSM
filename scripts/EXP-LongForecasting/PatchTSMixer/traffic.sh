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
dataset=traffic
num_channels=862

#Best configuration for traffic and 96 frames horizon
pred_len=96
python3 -u run_longExp.py \
  --train_epochs 15 \
  --activation 'gelu' \
  --dropout 0.3\
  --hidden_size 64\
  --num_blocks 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for traffic and 192 frames horizon
pred_len=192
python3 -u run_longExp.py \
  --train_epochs 15 \
  --activation 'gelu' \
  --dropout 0.3\
  --hidden_size 64\
  --num_blocks 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for traffic and 336 frames horizon
pred_len=336
python3 -u run_longExp.py \
  --train_epochs 15 \
  --activation 'gelu' \
  --dropout 0.3\
  --hidden_size 64\
  --num_blocks 6 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for traffic and 720 frames horizon
pred_len=720
python3 -u run_longExp.py \
  --train_epochs 15 \
  --activation 'gelu' \
  --dropout 0.6\
  --hidden_size 64\
  --num_blocks 8 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $num_channels \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 