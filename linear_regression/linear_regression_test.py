import numpy as np
import pandas as pd
import linear_regression as lin_reg

raw = pd.read_excel(('./data/kuiper.xls'))
y = np.array(raw.loc[:, 'Price'])

numeric_fields = ['Mileage', 'Cylinder', 'Liter']
boolean_fields = raw.loc[:, ['Cruise', 'Sound', 'Leather']]
almost_boolean =  ['Doors']
category_fields = ['Make', 'Model', 'Type']
# use t1.join(t2) when you get all the other columns ready

def ensure_df(x):
  if type(x) == pd.DataFrame:
    return x;
  else:
    return pd.DataFrame(x)

def fill_vals(item, base):
  base[item] = 1
  return base

def initialize_values(fields):
  res = {}
  for idx, val in enumerate(fields):
    res[val] = 0
  return res

def join_cols(*args):
  res = []
  for arg in args:
    if type(arg) is list:
      res.extend(arg)
    else:
      res.append(arg)

  return reduce(lambda accum, val: accum.join(val), map(ensure_df ,res))


category_columns = []
# map category columns
for idx, category in enumerate(category_fields):
  options = raw[category].unique()
  new_cols = raw[category].apply(lambda x: pd.Series(fill_vals(x, initialize_values(options))))
  category_columns.append(new_cols)

# map doors to boolean
doors_col = raw.loc[:, 'Doors'].apply(lambda x: int(x == 4))

numeric_cols = []
# map numeric columns to feature-scaled versions
for idx, numeric_col in enumerate(numeric_fields):
  base_col = raw.loc[:, numeric_col]
  std = base_col.std()
  mean = base_col.mean()
  scaled = base_col.apply(lambda x: (x - mean)/std)
  numeric_cols.append(scaled)

final_x = join_cols(category_columns, doors_col, numeric_cols, boolean_fields)

print 'resulting cost: {}, theta: {}'.format(lin_reg.run(final_x, y))
