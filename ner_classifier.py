import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer

from dataset import NER_Dataset, labels_dict, tokens_ids
from models import BERT_LSTM, BERT_LSTM_CRF


import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel
from torchmetrics import Precision, Recall, F1


class NER_Classifier(pl.LightningModule):
    
    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()
#             for key in classifier_instance.hparams.keys():
#                 self.hparams[key]=classifier_instance.hparams[key]
#                 print(key, classifier_instance.hparams[key])
            self.hparams = classifier_instance.hparams
            self.classifier = classifier_instance
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_type)
                    

        def train_dataloader(self) -> DataLoader:
            data_path_train = self.hparams.path_to_data + '/' + 'train.txt'
            
            self._train_dataset = NER_Dataset(data_path_train, self.tokenizer, 
                                              self.hparams.sequence_len, self.hparams.token_style)
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.batch_size, num_workers = self.hparams.loader_workers
            )

        def val_dataloader(self) -> DataLoader:
            data_path_val = self.hparams.path_to_data + '/' + 'val.txt'
            self._dev_dataset = NER_Dataset(data_path_val, self.tokenizer, 
                                              self.hparams.sequence_len, self.hparams.token_style)
            return DataLoader(
                dataset=self._dev_dataset,
                batch_size=self.hparams.batch_size, num_workers = self.hparams.loader_workers
            )

        def test_dataloader(self) -> DataLoader:
            data_path_test = self.hparams.path_to_data + '/' + 'test.txt'
            self._test_dataset = NER_Dataset(data_path_test, self.tokenizer, 
                                              self.hparams.sequence_len, self.hparams.token_style) 
            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.hparams.batch_size,num_workers = self.hparams.loader_workers
            )

    def __init__(self, hparams: Namespace) -> None:
        super(NER_Classifier, self).__init__()
        # save_hyperparameters https://discuss.pytorch.org/t/pytorch-lightning-module-cant-set-attribute-error/121125/5
        self.save_hyperparameters(hparams)
        print(self.hparams)
        
        self.batch_size = self.hparams.batch_size
        self.output_dim = len(self.hparams.features_dict)
        self.crf =  self.hparams.crf

        self.data = self.DataModule(self)

        self.__build_model()

        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_bert()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        #модель может использовать CRF слой, может не использовать
        self.model = BERT_LSTM(self.hparams.encoder_model)
        if self.crf:
            self.model = BERT_LSTM_CRF(self.hparams.encoder_model)

    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()
        
    def log_lh(self, x, attn_masks, y):
        #максимальное правдоподобие  - лос для CRF 
        ll = self.model.log_likelihood( x, attn_masks, y)
        return ll

    def unfreeze_bert(self) -> None:
        #разморозить слои bert
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.model.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_bert(self) -> None:
        #заморозить слои bert
        for param in self.model.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, batch: list) -> dict:
        #вход - батч (лист)
        #выход - словарь (target - таргет, predictions - предсказания модели)

        if self.training:
            self.eval()

        with torch.no_grad():
            if self.crf:
                model_out = self.forward(batch[0], batch[2], batch[1])
                model_out = model_out.view(-1)
            else:
                model_out = self.forward(batch[0], batch[2])
                model_out = model_out.view(-1, model_out.shape[2])
                predicted_labels = model_out.argmax(model_out, dim=1).view(-1)
                
        target = target.view(-1)
        pred_dict = {'target': target, 'predictions': predicted_labels}
        
        return pred_dict
    
    def forward(self, x, attn_masks, y):
        if self.crf:
            self.model.forward(x, attn_masks, y)
        else:
            self.model.forward(x, attn_masks)
        

    def loss(self, predictions, targets, batch = []) -> torch.tensor:

        if self.crf:
            return self.log_lh(batch[0], batch[2], batch[1])
        else:
            return self._loss(predictions, targets)

 
    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        #шаг обучения, на входе батч и номер батча
        #возвращает loss и инфу для логгера pl

        x, targets, attn_mask, y_mask = batch
        if self.crf:
            model_out = self.model.forward(x, attn_mask, targets)
            loss_val = self.log_lh(x, attn_mask, targets)
            predictions = model_out.view(-1)
            targets = targets.view(-1)
        else:
            model_out = self.model.forward(x, attn_mask)
            loss_val = self.loss(model_out.view(-1, model_out.shape[2]), targets.view(-1))

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        
        self.log('train_loss', loss_val, on_epoch=True, logger=True)      

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch: list, batch_nb: int, *args, **kwargs) -> dict:
        #шаг валидации - тоже, что и на обучении + считает accuracy на валидации
        num_classes = len(self.hparams.features_dict)
        x, targets, attn_mask, y_mask = batch
        
        precision_macro = Precision(average = 'macro', num_classes = num_classes).to(targets.device)
        recall_macro = Recall(average = 'macro', num_classes = num_classes).to(targets.device)
        f1_macro = F1(average = 'macro', num_classes = num_classes).to(targets.device)
  
        if self.crf:
            model_out = self.model.forward(x, attn_mask, targets)
            loss_val = self.log_lh(x, attn_mask, targets)
            predicted_labels = model_out.view(-1)
            targets = targets.view(-1)
        else:
            model_out = self.model.forward(x, attn_mask)
            loss_val = self.loss(model_out.view(-1, model_out.shape[2]), targets.view(-1))
            predicted_labels = torch.argmax(model_out.view(-1, model_out.shape[2]), dim = int(1)).view(-1)
            targets = targets.view(-1)

        # acc
        val_acc = torch.sum(targets == predicted_labels).item() / (len(targets) * 1.0)
        val_acc = torch.tensor(val_acc)
        
        val_precision = precision_macro(targets, predicted_labels)
        val_recall = recall_macro(targets, predicted_labels)
        val_f1 = f1_macro(targets, predicted_labels)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)
        
        self.log('val_loss', loss_val, on_epoch=True, logger=True)      
        self.log('val_acc', val_acc, on_epoch=True, logger=True)
        self.log('val_precision', val_precision, on_epoch=True, logger=True)  
        self.log('val_recall', val_recall, on_epoch=True, logger=True)  
        self.log('val_f1', val_f1, on_epoch=True, logger=True)  


        
        tqdm_dict = {"val_loss": loss_val,
                     "val_acc": val_acc, 'val_precision': val_precision , 'val_recall': val_recall,
                    "val_f1": val_f1}
        
        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,
                              'val_precision': val_precision, 'val_recall': val_recall,
                              "val_f1": val_f1,
                             "progress_bar": tqdm_dict, "log": tqdm_dict})

        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        #считает model performance для для ранней остановки

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]

            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            val_acc = output["val_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "val_acc": val_acc_mean,
            "val_loss": val_loss_mean
        }
        return result
    
    def test_step(self, batch: list, batch_nb: int, *args, **kwargs) -> dict:
        num_classes = len(self.hparams.features_dict)
        x, targets, attn_mask, y_mask = batch

        precision_macro = Precision(average = 'macro', num_classes = num_classes).to(targets.device)
        recall_macro = Recall(average = 'macro', num_classes = num_classes).to(targets.device)
        f1_macro = F1(average = 'macro', num_classes = num_classes).to(targets.device)
        
        precision_per_class = Precision(average = None, num_classes = num_classes).to(targets.device)
        recall_per_class = Recall(average = None, num_classes = num_classes).to(targets.device)
        f1_per_class = F1(average = None, num_classes = num_classes).to(targets.device)
        
        
        if self.crf:
            model_out = self.model.forward(x, attn_mask, targets)
            predicted_labels = model_out.view(-1)
            targets = targets.view(-1)
        else:
            model_out = self.model.forward(x, attn_mask)
            loss_val = self.loss(model_out.view(-1, model_out.shape[2]), targets.view(-1))
            predicted_labels = torch.argmax(model_out.view(-1, model_out.shape[2]), dim = int(1)).view(-1)
            targets = targets.view(-1)

        precision_m = precision_macro(targets, predicted_labels)
        recall_m = recall_macro(targets, predicted_labels)
        f1_m = f1_macro(targets, predicted_labels)
        print('precision_m', precision_m)
        print('recall_m', recall_m)
        print('f1_m', f1_m)
        
        precision_cl = precision_per_class(targets, predicted_labels)
        recall_cl = recall_per_class(targets, predicted_labels)
        f1_cl = f1_per_class(targets, predicted_labels)
        print('precision_cl', precision_cl)
        print('recall_cl', recall_cl)
        print('f1_cl', f1_cl)
        

        output = OrderedDict({"test_f1": f1_m, "test_precision": precision_m, "test_recall": recall_m})

        return output
        
    
    def test_epoch_end(self, outputs: list) -> dict:
        precision_mean = 0
        recall_mean = 0
        f1_mean = 0
        for output in outputs:
            precision = output["test_precision"]
            recall = output["test_recall"]
            f1 = output["test_f1"]

            if self.trainer.use_dp or self.trainer.use_ddp2:
                precision = torch.mean(precision)
            precision_mean += precision

            if self.trainer.use_dp or self.trainer.use_ddp2:
                recall = torch.mean(recall)
            recall_mean += recall
            
            if self.trainer.use_dp or self.trainer.use_ddp2:
                f1 = torch.mean(f1)
            f1_mean += f1



        precision_mean /= len(outputs)
        recall_mean /= len(outputs)
        f1_mean /= len(outputs)
        
        tqdm_dict = {"test_precision": precision_mean, "test_recall": recall_mean, "test_f1": f1_mean}
        
        result = {
            "progress_bar": tqdm_dict,
            "test_precision": precision_mean,
            "test_recall": recall_mean,
            "test_f1": f1_mean
        }
        return result       

    def configure_optimizers(self):
        parameters = [
            {
                "params": self.model.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams['learning_rate'])
        return [optimizer], []

    def on_epoch_end(self):
        #хук от лайтнинга
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_bert()
    
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        
        parser.add_argument(
            "--path_to_data",
            default = "./data/data_3_sm",
            type = str,
            help = " Путь до папки с 3 файлами: train.txt, val.txt, test.txt",
        )
        parser.add_argument(
            "--encoder_model",
            default = "deeppavlov_bert",
            type = str,
            help = "Энкодер, который используется - значения из словаря",
        )
        parser.add_argument(
            "--crf",
            default = False,
            type = bool,
            help = "Использование CRF слоя, default - False",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default = 1e-05,
            type = float,
            help = "learning rate для энкодера",
        )
        parser.add_argument(
            "--learning_rate",
            default = 3e-05,
            type = float,
            help = "learning rate для головы",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default = 1,
            type = int,
            help = "Количество эпох, на которые замораживаются слои энкодера, default = 1",
        )
        parser.add_argument(
            "--tokenizer_type",
            default = "DeepPavlov/rubert-base-cased",
            type = str,
            help = "Тип токенизатора - из словаря",
        )
        parser.add_argument(
            "--token_style",
            default = "bert",
            type = str,
            help = "Для расстановки специальных токенов в датасете, default - bert",
        )
        parser.add_argument(
            "--sequence_len",
            default = 256,
            type = np.int64,
            help = "Максимальная длина последовательности, default - 256",
        )

        parser.add_argument(
            "--loader_workers",
            default = 1,
            type = int,
            help = "Количество воркеров",
        )

        parser.add_argument(
            "--features_dict",
            default = labels_dict,
            type = dict,
            help = "Словарь с метками классов",
        )


        return parser