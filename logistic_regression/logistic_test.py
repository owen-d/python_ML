import pandas as pd
import numpy as np
from logistic import *

csv_input = pd.read_csv('./iris/Iris.csv')

x = csv_input.loc[:, 'SepalLengthCm':'PetalWidthCm']
x2 = csv_input.loc[:, 'SepalLengthCm':'SepalWidthCm']
y = csv_input['Species'].map(lambda x: int(x == 'Iris-setosa'))


# x2.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
# run our shit
run(x, y)
