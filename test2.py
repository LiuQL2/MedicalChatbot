import argparse

l = [200,300]

for s in l:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=s, help="the number of simulate epoch.")
    args = parser.parse_args()
    parameter = vars(args)
    print(parameter.get("simulate_epoch_number"))