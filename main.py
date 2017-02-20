import kagglegym
import numpy as np
import pandas as pd
import bz2
import base64
import pickle as pk
import warnings

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

# The "environment" is our interface.
env = kagglegym.make()

# We get our initial observation by calling "reset".
o = env.reset()

excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]
train = o.train.loc[:, col]

# Total number of NA values per observation.
train_NA_values = train.isnull().sum(axis=1)
    
# Record NA values and then fill them with the median.
d_mean = train.median(axis=0)

for c in col:
    train.loc[:, c + "_nan"] = pd.isnull(train[c])
    d_mean[c + "_nan"] = 0
    
train = train.fillna(d_mean)

train.loc[:, "is_null"] = train_NA_values

# Add mask with best features selection
model_2_mask = [ True, False, False,  True, False,  True, False,  True,  True,
       False, False,  True, False, False,  True,  True,  True,  True,
       False,  True, False,  True,  True,  True,  True, False,  True,
        True, False,  True, False,  True, False, False,  True,  True,
       False,  True,  True, False,  True, False, False,  True,  True,
       False,  True,  True, False,  True,  True,  True,  True,  True,
       False,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True, False,  True,  True, False, False,  True, False,
        True,  True, False, False, False,  True,  True, False,  True,
        True,  True, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True, False,  True,  True,  True,
        True, False, False, False,  True,  True,  True,  True, False,
       False, False, False, False, False, False, False, False,  True,
        True, False, False,  True,  True, False, False, False,  True,
       False, False,  True,  True, False, False, False, False,  True,
        True, False,  True, False,  True,  True,  True, False,  True,
       False,  True, False,  True, False,  True, False,  True, False,
        True, False,  True,  True, False,  True,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False, False, False,  True, False, False, False, False,
        True, False,  True, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
       False, False,  True, False, False,  True, False, False, False,  True]
train = train.loc[:, train.columns[model_2_mask]]

low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model_1 = LinearRegression(n_jobs=-1)
model_1.fit(np.array(train.loc[y_is_within_cut, "technical_20"].values).reshape(-1, 1), 
            o.train.loc[y_is_within_cut, "y"])

# Fit an ExtraTreesRegressor
extra_trees = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, 
                                  random_state=17, verbose=0)
model_2 = extra_trees.fit(train, o.train["y"])

# Load saved pickle model.
#model_2_str = """

#"""
#warnings.simplefilter("ignore", UserWarning)
#model_2 = pk.loads(bz2.decompress(base64.standard_b64decode(model_2_str)), encoding="latin1")

train = []

ymean_dict = dict(o.train.groupby(["id"])["y"].median())

while True:
    test = o.features.loc[:, col]

    # Total number of NA values per observation.
    test_NA_values = test.isnull().sum(axis=1)
    
    # Fill NA values.
    for c in col:
        test.loc[:, c + "_nan"] = pd.isnull(test[c])

    test = test.fillna(d_mean)
    
    test.loc[:, "is_null"] = test_NA_values

    # Add mask with best features selection
    test = test.loc[:, test.columns[model_2_mask]]

    pred = o.target
    test_technical_20 = np.array(test["technical_20"].values).reshape(-1, 1)
    
    # Ponderation of the two models.
    pred["y"] = ((model_1.predict(test_technical_20).clip(low_y_cut, high_y_cut) * 0.35)
                 + (model_2.predict(test).clip(low_y_cut, high_y_cut) * 0.65))
    # Add the median of the target value by ID.
    pred["y"] = pred.apply(lambda x: 0.95 * x["y"] + 0.05 * ymean_dict[x["id"]] if x["id"] in ymean_dict else x["y"], axis=1)
    
    # The target values have 6 decimals in the training set.
    pred["y"] = [float(format(x, ".6f")) for x in pred["y"]]
    
    o, reward, done, info = env.step(pred)
    if done:
        print("Finished", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)