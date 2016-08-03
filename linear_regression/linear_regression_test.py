import numpy as np
import pandas as pd
import linear_regression as lin_reg
import math

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

  # # uncomment for second order vars
  # second_order_base = base_col.apply(lambda x: math.pow(x, 2))
  # so_std = second_order_base.std()
  # so_mean = second_order_base.mean()
  # so_final = second_order_base.apply(lambda x: (x - so_mean)/so_std)
  # so_final = so_final.rename(so_final.name+'-squared')
  # numeric_cols.append(so_final)

final_x = join_cols(category_columns, doors_col, numeric_cols, boolean_fields)

theta, cost, hyp = lin_reg.run(final_x, y, max_iterations=1000000)
abs_dif = np.absolute(hyp-y)

print 'resulting cost: {}, theta: {}'.format(cost, theta)
print 'max distance: {}, min distance: {}, avg distance: {}, std dev of distance: {}'.format(abs_dif.max(), abs_dif.min(), abs_dif.mean(), abs_dif.std())
