import os
import sys
# sys.path.append(os.getcwd())
os.environ["PROJECT_ROOT"] = "E:/solo_transcription/note_attrib_prediction"
# os.environ["PROJECT_ROOT"] = "E:/solo_transcription"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.pylogger import get_pylogger
from utils.utils import *
from dataset.dataset import SoloDatasetModule
from tokenizer.tokenizer import TabTokenizer, tokenizer_initializer
from models.hybrid_ctc_lit_module import HybridCTCLitModule
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
import hydra
from typing import List, Optional, Tuple

log = get_pylogger(__name__)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating tokenizer <{cfg.tokenizer._target_}>")
    tokenizer_initializer: TabTokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer, vocab, vocab_size = tokenizer_initializer.generate()

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.dataset,
        data_preprocess_cfg=cfg.data_preprocess,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )

    log.info(f"Instantiating lightning module <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, tokenizer=tokenizer, vocab=vocab, vocab_size=vocab_size
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    
    logger: List[WandbLogger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "tokenizer": tokenizer,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if cfg.get("train"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
        else:
            ckpt_path = cfg.get("ckpt_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../conf", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()