"""General module unit tests for the KWS Spotter."""

import warnings

import numpy as np
import pytest
from omegaconf import OmegaConf

from src import data

warnings.filterwarnings("ignore")

cfg = OmegaConf.load("./config_dir/config.yaml")


@pytest.fixture
def mfcc() -> np.ndarray:
    """
    Fixture function to convert audio file to MFCC features to be
    used as a global variable in multiple tests.
    """
    mfcc_features = data.convert_audio_to_mfcc(
        cfg.names.audio_file,
        cfg.params.n_mfcc,
        cfg.params.mfcc_length,
        cfg.params.sampling_rate,
    )
    return mfcc_features


def test_label_type() -> None:
    """
    Function to test the datatype of labels which should be
    `str` in order to be used for training and inference.
    """
    labels = data.Preprocess().wrap_labels()
    assert all(isinstance(n, str) for n in labels)


def test_mfcc_shape(mfcc: pytest.fixture) -> None:
    """
    Function to test the shape of computed MFCC features
    from audio files. It is an ndarray whose shape should
    match the parameters(n_mfcc, mfcc_length) from the config.
    """
    assert mfcc.shape == (cfg.params.n_mfcc, cfg.params.mfcc_length)


def test_mfcc_dimension(mfcc: pytest.fixture) -> None:
    """Function to test the dimension of mfcc features array."""
    assert len(mfcc.shape) == 2
