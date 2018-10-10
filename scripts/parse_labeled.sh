mkdir -p parsed_data/labeled
python -m ts.scripts.parse_mat data/beedance/beedance-1.mat data/beedance/beedance-2.mat data/beedance/beedance-3.mat data/beedance/beedance-4.mat data/beedance/beedance-5.mat data/beedance/beedance-6.mat parsed_data/labeled/beedance.npy
python -m ts.scripts.parse_mat data/fishkiller/fishkiller.mat parsed_data/labeled/fishkiller.npy
python -m ts.scripts.parse_mat data/hasc/hasc-1.mat parsed_data/labeled/hasc.npy
python -m ts.scripts.parse_mat \
  data/yahoo/yahoo-7.mat \
  data/yahoo/yahoo-8.mat \
  data/yahoo/yahoo-16.mat \
  data/yahoo/yahoo-22.mat \
  data/yahoo/yahoo-27.mat \
  data/yahoo/yahoo-33.mat \
  data/yahoo/yahoo-37.mat \
  data/yahoo/yahoo-42.mat \
  data/yahoo/yahoo-45.mat \
  data/yahoo/yahoo-46.mat \
  data/yahoo/yahoo-50.mat \
  data/yahoo/yahoo-51.mat \
  data/yahoo/yahoo-54.mat \
  data/yahoo/yahoo-55.mat \
  data/yahoo/yahoo-56.mat \
  parsed_data/labeled/yahoo.npy
