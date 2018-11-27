CUDA_VISIBLE_DEVICES=2 python -m ts.scripts.train_rnn --data-paths \
  parsed_data/unlabeled/exchange_rate.npy \
  parsed_data/unlabeled/electricity.npy \
  parsed_data/unlabeled/solar.npy \
  parsed_data/unlabeled/traffic.npy \
  --normal running \
  --model-type lstnet \
  --save-path models/bi-lstnet-r \
  --num-epochs 10 \
  --cuda \
  --batch-size 128
# CUDA_VISIBLE_DEVICES=  python -m ts.scripts.test_nab2 \
#   --model-path models/bi-lstnet
# CUDA_VISIBLE_DEVICES=  python -m ts.scripts.test_nab \
#   --model-path models/bi-lstnet
