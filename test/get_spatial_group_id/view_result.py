import pandas as pd

my_result = pd.read_csv('out.txt', sep=' ', header=None).to_numpy()
result = pd.read_csv('result.txt', sep=' ', header=None).to_numpy()

print(1)