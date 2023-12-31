"""
Defines and create model for training and evaluation.

DeepSpeech 2.0 architecture is used for this project with 1D convolutional layers
followed by LSTM layers with self-attention and fully connected layers. 
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from keras.layers import (LSTM, BatchNormalization, Conv1D, Dense, Dropout,
                          Input, MaxPooling1D)
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention


class Models(ABC):
    """
    Abstract base class that defines and creates model.
    """

    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass


@dataclass
class DeepSpeechModel(Models):
    """
    Dataclass to create CNN-LSTM model that inherits Models class.
    """

    input_shape: Tuple[int, int]
    num_classes: int

    def define_model(self) -> Sequential:
        """
        Method to define model that can be used for training
        and inference. This existing model can also be tweaked
        by changing parameters, based on the requirements.

        Parameters
        ----------
            None.

        Returns
        -------
        Sequential
        """

        return Sequential(
            [
                Input(shape=self.input_shape),
                BatchNormalization(),
                # 1D Convolutional layers
                Conv1D(32, kernel_size=3, strides=1, padding="same"),
                BatchNormalization(),
                MaxPooling1D(pool_size=3),
                Conv1D(64, kernel_size=3, strides=1, padding="same"),
                BatchNormalization(),
                MaxPooling1D(pool_size=3),
                Conv1D(128, kernel_size=3, strides=1, padding="same"),
                BatchNormalization(),
                MaxPooling1D(pool_size=3, padding="same"),
                Dropout(0.30),
                # LSTM layers
                LSTM(units=128, return_sequences=True),
                SeqSelfAttention(attention_activation="tanh"),
                LSTM(units=128, return_sequences=False),
                BatchNormalization(),
                Dropout(0.30),
                # Dense layers
                Dense(256, activation="relu"),
                Dense(64, activation="relu"),
                Dropout(0.30),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

    def create_model(self) -> Sequential:
        """
        Method to create the model defined by define_model() method
        and prints the model summary.

        Parameters
        ----------
            None.

        Returns
        -------
        model: Sequential
        """
        model: Sequential = self.define_model()
        model.summary()
        return model
