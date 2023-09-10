"""
Load audio dataset, preprocess them and train CNN-LSTM
model and log the metrics.
"""

import warnings

import hydra
from hydra.core.config_store import ConfigStore
from keras.models import Sequential

from config_dir.configType import KWSConfig
from src import data, train
from src.experiment_tracking import MLFlowTracker, ModelSelection
from src.model import DeepSpeechModel

warnings.filterwarnings("ignore")

cs = ConfigStore.instance()
cs.store(name="kws_config", node=KWSConfig)


@hydra.main(config_path="config_dir", config_name="config")
def main(cfg: KWSConfig) -> None:
    """Initializes the training pipeline."""
    try:
        # Initializing MLFlow for model tracking and logging
        tracker = MLFlowTracker(
            cfg.names.experiment_name, cfg.paths.mlflow_tracking_uri
        )
        tracker.log()

        # Load and preprocess the audio dataset for training
        dataset_ = data.Dataset()
        preprocess_ = data.Preprocess(
            dataset_,
            cfg.paths.train_dir,
            cfg.params.n_mfcc,
            cfg.params.mfcc_length,
            cfg.params.sampling_rate,
        )
        preprocessed_dataset: data.Dataset = preprocess_.preprocess_dataset(
            preprocess_.labels, cfg.params.test_data_split_percent
        )
        [
            data.print_shape(key, value)
            for key, value in preprocessed_dataset.__dict__.items()
        ]

        # Loading and training the model
        model: Sequential = DeepSpeechModel(
            (cfg.params.n_mfcc, cfg.params.mfcc_length), len(preprocess_.labels)
        ).create_model()
        best_selected_model: ModelSelection = train.Training(
            model,
            preprocessed_dataset,
            cfg.params.batch_size,
            cfg.params.epochs,
            cfg.params.learning_rate,
            tracker,
            cfg.names.metric_name,
        ).train()

    except Exception as exc:
        raise Exception("ffhffh") from exc


if __name__ == "__main__":
    main()
