full_train = pd.read_hdf("train.h5")
o_train, o_test = train_test_split(full_train, test_size = 0.5, random_state=0)

excl = ["id", "timestamp", "y"]
#excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o_train.columns if c not in excl]
train = o_train.loc[:, col]

# Total number of NA values per observation.
train_NA_values = train.isnull().sum(axis=1)

# Record NA values and then fill them with the median.
d_mean = train.median(axis=0)

for c in col:
    train.loc[:, c + "_nan"] = pd.isnull(train[c])
    d_mean[c + "_nan"] = 0
    
train = train.fillna(d_mean)

train.loc[:, "is_null"] = train_NA_values

model_2_mask = pd.read_csv("model_2_mask.csv", encoding="utf-8")
train = train.loc[:, train.columns[model_2_mask["mask"]]]

# By timestamp: std of target value thanks to the variable "technical_30" and count of assets.
#dict_y_std_timestamp = dict(np.sqrt(o_train.groupby("timestamp")["technical_30"].var()))
#train.loc[:, "y_std_timestamp"] = o_train["timestamp"].map(dict_y_std_timestamp)

#dict_assets_count = dict(o_train.groupby("timestamp")["id"].count())
#train.loc[:, "assets_count"] = o_train["timestamp"].map(dict_assets_count)

# Train model
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o_train.y > high_y_cut)
y_is_below_cut = (o_train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model_1 = LinearRegression(n_jobs=-1)
model_1.fit(np.array(train.loc[y_is_within_cut, "technical_20"].values).reshape(-1, 1), 
            o_train.loc[y_is_within_cut, "y"])


# Fit an ExtraTreesRegressor
t0 = time()

extra_trees = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, 
                                  random_state=17, verbose=0)
model_2 = extra_trees.fit(train, o_train["y"])

train = []

print("Running time: {0}s.".format(time() - t0))

#mask = model_2.feature_importances_ >= 0.001
#pd.DataFrame({"mask": mask}).to_csv("model_2_mask.csv", 
 #                                           encoding="utf-8",
                                            #index=False)

# A custom function to compute the R score
def get_reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o_train.groupby(["id"])["y"].median())

t0 = time()

test_timestamp = sorted(o_test.timestamp.unique())

y_predict = []
y_real = []

for timestp in [1,2]:
    o_test_current = o_test.loc[o_test["timestamp"] == timestp, :]
    
    test = o_test_current.loc[:, col]
    y_real.extend(o_test_current.loc[:, "y"])

    # Total number of NA values per observation.
    test_NA_values = test.isnull().sum(axis=1)

    # Fill NA values.
    for c in col:
        test.loc[:, c + "_nan"] = pd.isnull(test[c])

    test = test.fillna(d_mean)

    test.loc[:, "is_null"] = test_NA_values
    
    test = test.loc[:, test.columns[model_2_mask["mask"]]]

    # By timestamp: std of target value thanks to the variable "technical_30" and count of assets.
    #dict_y_std_timestamp = dict(np.sqrt(o_test.groupby("timestamp")["technical_30"].var()))
    #test.loc[:, "y_std_timestamp"] = o_test["timestamp"].map(dict_y_std_timestamp)

    #dict_assets_count = dict(o_test.groupby("timestamp")["id"].count())
    #test.loc[:, "assets_count"] = o_test["timestamp"].map(dict_assets_count)

    pred = o_test_current.loc[:, ["id", "y"]]
    test_technical_20 = np.array(test["technical_20"].values).reshape(-1, 1)

    # Ponderation of the two models.
    pred.loc[:, "y"] = ((model_1.predict(test_technical_20).clip(low_y_cut, high_y_cut) * 0.35)
                        + (model_2.predict(test).clip(low_y_cut, high_y_cut) * 0.65))
    # Add the median of the target value by ID.
    pred.loc[:, "y"] = pred.apply(lambda x: 0.95 * x["y"] + 0.05 * ymean_dict[x["id"]]
                                  if x["id"] in ymean_dict 
                                  else x["y"], axis=1)

    # The target values have 6 decimals in the training set.
    pred.loc[:, "y"] = [float(format(x, ".6f")) for x in pred.loc[:, "y"]]
    
    y_predict.extend(pred["y"])

LB_score = get_reward(np.array(y_real), np.array(y_predict))

print("Finished", LB_score)

print("Running time: {0}s.".format(time() - t0))


# Base model 0.021145763244338721 
# with >0.001 features selection 0.022024814143177515 

df_y_plot = o_test.loc[:, ["timestamp"]]
df_y_plot.loc[:, "y"] = np.array(y_predict)

plt.figure(figsize=(12, 6))
plt.plot(o_test.groupby("timestamp")["y"].var(), color="blue")
plt.plot(df_y_plot.groupby("timestamp")["y"].var(), color="red")
plt.xlim(0, 500)
plt.ylim(0, 0.001)
plt.show()