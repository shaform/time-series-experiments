python -m ts.scripts.eval_wavenet \
  --cuda \
  --label beedance \
  --finetune \
  --data-paths \
  parsed_data/labeled/beedance-1.npy \
  parsed_data/labeled/beedance-2.npy \
  parsed_data/labeled/beedance-3.npy \
  parsed_data/labeled/beedance-4.npy \
  parsed_data/labeled/beedance-5.npy \
  parsed_data/labeled/beedance-6.npy >> results_wave.txt
python -m ts.scripts.eval_wavenet \
  --cuda \
  --label fishkiller \
  --finetune \
  --data-paths \
  parsed_data/labeled/fishkiller.npy >> results_wave.txt
python -m ts.scripts.eval_wavenet \
  --cuda \
  --label hasc \
  --finetune \
  --data-paths \
  parsed_data/labeled/hasc.npy >> results_wave.txt
python -m ts.scripts.eval_wavenet \
  --cuda \
  --label yahoo \
  --finetune \
  --data-paths \
  parsed_data/labeled/yahoo-22.npy \
  parsed_data/labeled/yahoo-7.npy \
  parsed_data/labeled/yahoo-8.npy \
  parsed_data/labeled/yahoo-16.npy \
  parsed_data/labeled/yahoo-22.npy \
  parsed_data/labeled/yahoo-27.npy \
  parsed_data/labeled/yahoo-33.npy \
  parsed_data/labeled/yahoo-37.npy \
  parsed_data/labeled/yahoo-42.npy \
  parsed_data/labeled/yahoo-45.npy \
  parsed_data/labeled/yahoo-46.npy \
  parsed_data/labeled/yahoo-50.npy \
  parsed_data/labeled/yahoo-51.npy \
  parsed_data/labeled/yahoo-54.npy \
  parsed_data/labeled/yahoo-55.npy \
  parsed_data/labeled/yahoo-56.npy >> results_wave.txt
