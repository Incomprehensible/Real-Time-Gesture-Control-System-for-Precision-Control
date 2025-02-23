import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Visualize data from uMyo")
parser.add_argument(
    "--file", type=str, default="gilbert_raw.csv", help="File to visualize"
)
args = parser.parse_args()

df = pd.read_csv(args.file, header=None)
data = np.array(df).flatten()

print(f'Max value: {max(data)}')
print(f'Min value: {min(data)}')

plt.title('Myoware data')
plt.plot(data)
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()
