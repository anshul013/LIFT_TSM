if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TSMixer" ]; then
    mkdir ./logs/LongForecasting/TSMixer
fi
seq_len=336
model_name=TSMixer
dataset=weather
num_channels=21

#Best configuration for weather and 96 frames horizon
pred_len=96
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.3\
  --hidden_size 32\
  --num_blocks 4 \
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
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for weather and 192 frames horizon
pred_len=192
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.7\
  --hidden_size 8\
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
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for weather and 336 frames horizon
pred_len=336
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.7\
  --hidden_size 8\
  --num_blocks 4 \
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
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 

#Best configuration for weather and 720 frames horizon
pred_len=720
python3 -u run_longExp.py \
  --activation 'relu' \
  --dropout 0.7\
  --hidden_size 64\
  --num_blocks 4 \
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
  --patience 5 \
  --train_epochs 100 \
  --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'/'$dataset'_'$seq_len'_'$pred_len.log 