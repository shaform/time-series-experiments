for name in fishkiller hasc beedance-1 beedance-2 beedance-3 beedance-4 beedance-5 beedance-6 yahoo-7 yahoo-8 yahoo-16 yahoo-22 yahoo-27 yahoo-33 yahoo-37 yahoo-42 yahoo-45 yahoo-46 yahoo-50 yahoo-51 yahoo-54 yahoo-55 yahoo-56 ; do
  python -m ts.scripts.train_cpd \
    --data-path parsed_data/labeled/$name.npy \
    --cuda \
    --bidirectional \
    --predict-horizon \
    --save-path models/cpd/bi-h/$name
done
