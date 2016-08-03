import numpy as np
import pandas as pd

raw = pd.read_excel(('./data/kuiper.xls'))
y = np.array(raw.loc[:, 'Price'])

x1 = raw.loc[:, ['Mileage', 'Cylinder', 'Liter', 'Doors', 'Cruise', 'Sound', 'Leather']]
category_fields = ['Make', 'Model', 'Type']
# use t1.join(t2) when you get all the other columns ready

def fill_vals(item, base):
  base[item] = 1
  return base

def initialize_values(fields):
  res = {}
  for idx, val in enumerate(fields):
    res[val] = 0
  return res


category_columns = []
for idx, category in enumerate(category_fields):
  options = raw[category].unique()
  new_cols = raw[category].apply(lambda x: pd.Series(fill_vals(x, initialize_values(options))))
  category_columns.append(new_cols)

final_x = x1.join(category_columns)

print final_x.head(), final_x.shape

