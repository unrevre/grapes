# pylint: disable=missing-docstring,invalid-name

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import (
    activations,
    layers,
    losses,
    metrics,
    optimizers,
    regularizers,
)


class TensorflowNetwork:
    def __init__(self, **kwargs):
        groups = (('model',), ('size', 'depth'))

        if all(k in kwargs for k in groups[0]):
            args = [kwargs[k] for k in groups[0]]
            self.model = keras.models.load_model(*args)
            return

        if all(k in kwargs for k in groups[1]):
            args = [kwargs[k] for k in groups[1]]
            self.model = self.build(*args)
            return

        raise KeyError

    def build(self, size, depth):
        inputs = keras.Input(shape=(size, size, depth * 2 + 1))

        x = layers.Conv2D(
            8,
            3,
            padding="same",
            data_format="channels_last",
            kernel_regularizer=regularizers.L2(1e-4),
            name='conv',
        )(inputs)
        x = layers.BatchNormalization(axis=1, name='n')(x)
        x = layers.Activation(activations.relu, name='a')(x)

        for i in range(1):
            y = x

            x = layers.Conv2D(
                8,
                3,
                padding="same",
                data_format="channels_last",
                kernel_regularizer=regularizers.L2(1e-4),
                name='res_conv0{}'.format(i),
            )(x)
            x = layers.BatchNormalization(axis=1, name='n0{}'.format(i))(x)
            x = layers.Activation(activations.relu, name='a0{}'.format(i))(x)
            x = layers.Conv2D(
                8,
                3,
                padding="same",
                data_format="channels_last",
                kernel_regularizer=regularizers.L2(1e-4),
                name='res_conv1{}'.format(i),
            )(x)
            x = layers.BatchNormalization(axis=1, name='n1{}'.format(i))(x)
            x = layers.Add(name='add{}'.format(i))([x, y])
            x = layers.Activation(activations.relu, name='a1{}'.format(i))(x)

        p = layers.Conv2D(
            8,
            1,
            data_format="channels_last",
            kernel_regularizer=regularizers.L2(1e-4),
            name='policy_conv',
        )(x)
        p = layers.BatchNormalization(axis=1, name='np')(p)
        p = layers.Activation(activations.relu, name='ap')(p)
        p = layers.Flatten(name='flatp')(p)
        p = layers.Dense(
            size * size + 1,
            kernel_regularizer=regularizers.L2(1e-4),
            name='policy_dense',
        )(p)
        p = layers.Activation(activations.softmax, name='softmax')(p)

        v = layers.Conv2D(
            4,
            1,
            data_format="channels_last",
            kernel_regularizer=regularizers.L2(1e-4),
            name='value_conv',
        )(x)
        v = layers.BatchNormalization(axis=1, name='nv')(v)
        v = layers.Activation(activations.relu, name='av')(v)
        v = layers.Flatten(name='flatv')(v)
        v = layers.Dense(
            1,
            kernel_regularizer=regularizers.L2(1e-4),
            name='value_dense',
        )(v)
        v = layers.Activation(activations.tanh, name='tanh')(v)

        return keras.Model(inputs=inputs, outputs=[p, v])

    def compile(self, bounds, values, momentum):
        self.model.compile(
            optimizer=optimizers.SGD(
                learning_rate=optimizers.schedules.PiecewiseConstantDecay(
                    boundaries=bounds, values=values
                ),
                momentum=momentum,
            ),
            loss=[losses.CategoricalCrossentropy(), losses.MeanSquaredError()],
            metrics=[
                [metrics.MeanSquaredError(), metrics.CategoricalAccuracy()],
                [metrics.MeanSquaredError()],
            ],
        )

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, **kwargs)

    def eval(self, inputs):
        return self.model(inputs, training=False)

    def save(self, path):
        return self.model.save(path)
