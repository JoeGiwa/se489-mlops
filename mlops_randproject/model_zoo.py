import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    Input,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
)


def build_mlp(cfg, input_shape):
    model = Sequential(
        [
            Input(shape=(input_shape,)),
            Dense(cfg.hidden1, activation="relu", kernel_regularizer=l2(cfg.l2)),
            BatchNormalization(),
            Dropout(cfg.dropout),
            Dense(cfg.hidden2, activation="relu", kernel_regularizer=l2(cfg.l2)),
            BatchNormalization(),
            Dropout(cfg.dropout),
            Dense(cfg.output, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=cfg.optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(cfg, input_shape):
    model = Sequential()
    model.add(
        Conv1D(
            filters=cfg.filters[0],
            kernel_size=cfg.kernel_size,
            activation="relu",
            input_shape=(input_shape, 1),
        )
    )
    model.add(MaxPooling1D(pool_size=cfg.pool_size))
    model.add(Dropout(cfg.dropout))
    model.add(BatchNormalization())

    if cfg.conv_layers > 1:
        model.add(
            Conv1D(
                filters=cfg.filters[1], kernel_size=cfg.kernel_size, activation="relu"
            )
        )
        model.add(MaxPooling1D(pool_size=cfg.pool_size))
        model.add(Dropout(cfg.dropout))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(cfg.dense_units, activation="relu"))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.output, activation="softmax"))

    model.compile(
        optimizer=cfg.optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_xgboost(cfg):
    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="multi:softmax",
        num_class=cfg.output,
        seed=cfg.random_state,
    )
    return model
