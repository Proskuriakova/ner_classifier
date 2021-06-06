import argparse
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, CSVLogger

from ner_classifier import NER_Classifier

#запуск модели
def main(hparams) -> None:

#     # Инициализируем модель
    model = NER_Classifier(hparams)
    
    #  Модуль для раннего останова
    
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta = 0.01,
        patience = hparams.patience,
        verbose = True,
        mode = hparams.metric_mode,
    )

    # Логгер для тензорборда

    tb_logger = TensorBoardLogger(
        save_dir="./experiments",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )
    
    ckpt_path = os.path.join(
        "./experiments/", tb_logger.version, "checkpoints",
    )
    
    csv_logger = CSVLogger('./', name='csv_logs', version='commadot_sm')

    # Чекпоинты
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k = hparams.save_top_k,
        verbose = True,
        monitor = hparams.monitor,
        period = 1,
        mode = hparams.metric_mode,
        save_weights_only = True
    )

    # Инициализация pl класса трэйнера
    #callbacks= early_stop_callback,

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        checkpoint_callback=True,
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
        distributed_backend="dp",
    )
    
#     #  запуск обучения
    print('START TRAINING')
    trainer.fit(model, model.data)
    print('START TESTING')
#     chk_path = "./experiments/version_01-06-2021--18-09-58/checkpoints/epoch=2-step=315.ckpt"
#     model_saved = model.load_from_checkpoint(chk_path)
    trainer.test(model, datamodule = model.data, verbose=True)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=20,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default = 2.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )

    # each LightningModule defines arguments relevant to it
    parser = NER_Classifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
    

