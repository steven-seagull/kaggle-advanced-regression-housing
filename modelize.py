# %%
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
import numpy as np

house_prices_test_df = pd.read_csv(
    "datasets/test.csv")

house_prices_train_df = pd.read_csv(
    "datasets/train.csv")

house_prices_test_df.drop(["Id"], axis = 1, inplace = True)
house_prices_train_df.drop(["Id"], axis = 1, inplace = True)

y = house_prices_train_df.pop("SalePrice")
x = house_prices_train_df
test_x = house_prices_test_df[x.columns]



transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"),
    StandardScaler(),
)
preprocessor = make_column_transformer(
    (transformer_num,
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False, handle_unknown="ignore"),
     make_column_selector(dtype_include=object)),
)

x = preprocessor.fit_transform(x)

y = np.log(y)

input_shape = [x.shape[1]]


early_stopping = EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=20,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mae"
)

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

fitted_model = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=250,
    epochs=256,
    callbacks=[early_stopping]
)

test_x = preprocessor.transform(test_x)
print(x.shape)
print(test_x.shape)
predictions = model.predict(test_x)
predictions = np.exp(predictions)

house_prices_test_df = pd.read_csv(
    "datasets/test.csv")
house_prices_test_df["SalePrice"] = predictions
house_prices_test_df.set_index("Id", inplace=True)
house_prices_test_df[["Id", "SalePrice"]].to_csv("submission.csv")
history_df = pd.DataFrame(fitted_model.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
