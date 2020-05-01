import torch
import torch.nn as nn
import transformers
import os
import logging
import json

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from collections import OrderedDict

from typing import List, Dict, Tuple, Iterable, Type
from torch.optim import Optimizer
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import import_from_string, batch_to_device, http_get

## import tensorboard.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils import *

# setup logger
logger = init_logger(__name__)

## inherit SentenceTransformer here
class SentenceTransformer_tb(SentenceTransformer):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, 
                saved_scibert_model_path: str = None,
                device: str = None):

        if saved_scibert_model_path == None:
            super().__init__(model_name_or_path, modules, device)
        else:
            logging.info("Load SentenceTransformer from folder: {}".format(saved_scibert_model_path))
            if os.path.exists(os.path.join(saved_scibert_model_path, 'config.json')):
                with open(os.path.join(saved_scibert_model_path, 'config.json')) as fIn:
                    config = json.load(fIn)

            with open(os.path.join(saved_scibert_model_path, 'modules.json')) as fIn:
                contained_modules = json.load(fIn)

            modules = OrderedDict()
            for module_config in contained_modules:
                if '__main__' in module_config['type']:
                    module_config_real_type = "sentence_transformers.models.BERT"
                    module_class = import_from_string(module_config_real_type)
                else:
                    module_class = import_from_string(module_config['type'])
                loading_path = os.path.join(saved_scibert_model_path, module_config['path'])
                module = module_class.load(os.path.join(saved_scibert_model_path, module_config['path']))
                modules[module_config['name']] = module

            super(SentenceTransformer, self).__init__(modules)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logging.info("Use pytorch device: {}".format(device))
            self.device = torch.device(device)
            self.to(device)

        self.desc_string = 'tb_'

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            eval_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            train_evaluator: SentenceEvaluator,
            evaluator: SentenceEvaluator,
            train_phase: str = 'STS',
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object ]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        
        if train_phase not in ['STS', 'NLI']:
            assert False, print(f"Not valid train_phase given.")

        self.lr = optimizer_params['lr']
        self.desc_string = f'{train_phase}-lr-{self.lr}_epochs-{epochs}_warmup_steps-{warmup_steps}'
        print(f"model description is {self.desc_string}.")

        if output_path is not None:
            # empty folder is not necessary.
            os.makedirs(output_path, exist_ok=True)
            path_prefix = output_path.split('/')[0]
            tb_writer = SummaryWriter(log_dir=os.path.join(path_prefix, self.desc_string))
            tb_writer.add_text('experiment args', self.desc_string, 0)
            
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        ## GX: this design is for the composite loss.
        loss_models = [loss for _, loss in train_objectives]
        eval_dataloader, evaluation_loss = eval_objectives[0]   # the current version
        device = self.device

        for loss_model in loss_models:
            loss_model.to(device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers w.r.t each model.
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = num_train_steps
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
            
        # Decides the data-type here.
        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(loss_models)):
                model, optimizer = amp.initialize(loss_models[train_idx], optimizers[train_idx], opt_level=fp16_opt_level)
                loss_models[train_idx] = model
                optimizers[train_idx] = optimizer

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)

        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]   

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self.device)
                    loss_value = loss_model(features, labels)
                    tb_writer.add_scalar("progress/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("progress/steps_per_epoch", steps_per_epoch, global_step)
                    tb_writer.add_scalar("progress/num_train_steps", num_train_steps, global_step)                                
                    tb_writer.add_scalar("train/loss_value", loss_value, global_step)

                    # task specific:
                    criterion = train_evaluator(self, output_path=output_path, epoch=epoch, steps=global_step) 
                    if train_phase == 'NLI':
                        tb_writer.add_scalar("train/accuracy", criterion, global_step)
                    elif train_phase == 'STS':
                        tb_writer.add_scalar("train/score", criterion, global_step)

                    if fp16:
                        with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_loss(evaluation_loss, eval_dataloader, tb_writer, global_step)
                    self._eval_during_training_custom(evaluator, output_path, tb_writer, save_best_model, epoch, 
                                                    training_steps, global_step)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_loss(evaluation_loss, eval_dataloader, tb_writer, global_step)
            self._eval_during_training_custom(evaluator, output_path, tb_writer, save_best_model, epoch, -1, global_step)

    def _eval_loss(self, evaluation_loss, eval_dataloader, tb_writer, global_step):
        """evalution on cos similarity w.r.t STS."""

        eval_dataloader.collate_fn = self.smart_batching_collate
        loss_value = 0
        with torch.no_grad():
            for idx, batch_cur in enumerate(eval_dataloader):
                features, labels = batch_to_device(batch_cur, self.device)
                loss_value += evaluation_loss(features, labels)

        loss_value_per_batch = float(loss_value/idx)
        tb_writer.add_scalar("eval/loss", loss_value_per_batch, global_step)

    def _eval_during_training_custom(self, evaluator, output_path, tb_writer, save_best_model, epoch, steps, global_step):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            tb_writer.add_scalar("eval/score", score, global_step)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score
