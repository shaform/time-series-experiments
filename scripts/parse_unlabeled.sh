mkdir -p parsed_data/unlabeled
python -m ts.scripts.parse_csv data/electricity/electricity.txt parsed_data/unlabeled/electricity.npy
python -m ts.scripts.parse_csv data/exchange_rate/exchange_rate.txt parsed_data/unlabeled/exchange_rate.npy
python -m ts.scripts.parse_csv data/solar-energy/solar_AL.txt parsed_data/unlabeled/solar.npy
python -m ts.scripts.parse_csv data/traffic/traffic.txt parsed_data/unlabeled/traffic.npy
