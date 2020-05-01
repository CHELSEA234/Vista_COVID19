import torch
import torch.nn as nn
import transformers
import os
import json

from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from typing import List, Dict, Tuple, Iterable, Type
from torch.optim import Optimizer
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import import_from_string, batch_to_device, http_get
from utils import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# setup logger
logger = init_logger(__name__)
 
## inherit SentenceTransformer here
class SentenceTransformer_tb(SentenceTransformer):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, 
                saved_scibert_model_path: str = None,
                device: str = None):

        self.loading_model_dir = saved_scibert_model_path
        if saved_scibert_model_path == None:
            super().__init__(model_name_or_path, modules, device)
        else:
            logger.info("Load SentenceTransformer from folder: {}".format(saved_scibert_model_path))
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
                logger.info("Use pytorch device: {}".format(device))
            self.device = torch.device(device)
            self.to(device)

        self.desc_string = 'tb_'

    def save(self, file_name, args, optimizer, scheduler):
        '''save everything here.'''
        current_output_dir = os.path.join(args.output_dir, file_name)
        super().save(current_output_dir)  # saves the model
        torch.save(args, os.path.join(current_output_dir, "training_config.bin"))         # save the config
        torch.save(optimizer.state_dict(), os.path.join(current_output_dir, "optimizer.pt"))    # save the optimizer
        torch.save(scheduler.state_dict(), os.path.join(current_output_dir, "scheduler.pt"))    # save the scheduler

    def load(self, current_output_dir, steps_per_epoch):
        '''loading the pretrained weights.'''
        args_dict = vars(torch.load(os.path.join(current_output_dir, "training_config.bin")))
        global_step = args_dict['global_step']
        epochs_trained = global_step // steps_per_epoch
        steps_trained_in_current_epoch = global_step % steps_per_epoch

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        return global_step, epochs_trained, steps_trained_in_current_epoch

    def fit(self,
            args,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            train_evaluator: SentenceEvaluator,
            evaluation_loss: nn.Module,
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            eval_dataloader = None,
            steps_per_epoch = None,
            scheduler_name: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object ]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        
        self.lr = optimizer_params['lr']
        self.desc_string = f'lr-{self.lr}_epochs-{epochs}_warmup_steps-{warmup_steps}'
        logger.info(f"model description is {self.desc_string}.")
        args.desc_string = self.desc_string

        if args.output_dir is not None:
            # empty folder is not necessary.
            os.makedirs(args.output_dir, exist_ok=True)
            path_prefix = args.output_dir.split('/')[0]
            tb_writer = SummaryWriter(log_dir=os.path.join(path_prefix, self.desc_string))
            tb_writer.add_text('experiment args', self.desc_string, 0)
            
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        ## GX: this design is for the composite loss.
        loss_model = [loss for _, loss in train_objectives][0]
        device = self.device
        loss_model.to(device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizer and schedule (linear warmup and decay)
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
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler_name, warmup_steps=warmup_steps, t_total=t_total)

        # Config
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        # Check if continuing training from a checkpoint
        if self.loading_model_dir and os.path.exists(self.loading_model_dir):
            optimizer.load_state_dict(torch.load(os.path.join(self.loading_model_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.loading_model_dir, "scheduler.pt")))
            global_step, epochs_trained, steps_trained_in_current_epoch = self.load(self.loading_model_dir, steps_per_epoch)
        else:
            logger.info("  Starting fine-tuning.")

        # Train !  
        for epoch in trange(epochs_trained, epochs, desc="Epoch"):
            training_steps = 0  # training steps per epoch.
            loss_model.zero_grad()
            loss_model.train()
            data_iterator = data_iterators[0] 
            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloaders[0])
                    data_iterators[0] = data_iterator
                    data = next(data_iterator)

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    training_steps += 1
                    continue

                features, labels = batch_to_device(data, self.device)
                loss_value = loss_model(features, labels)
                tb_writer.add_scalar("progress/lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("progress/steps_per_epoch", steps_per_epoch, global_step)
                tb_writer.add_scalar("progress/num_train_steps", num_train_steps, global_step)
                tb_writer.add_scalar("train/loss_value", loss_value, global_step)

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

                global_step += 1
                training_steps += 1
                args.global_step = global_step

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    accuracy = train_evaluator(self, output_path=args.output_dir, epoch=epoch, steps=training_steps)
                    tb_writer.add_scalar("train/accuracy", accuracy, global_step)
                    current_file_name = 'checkpoint-'+str(global_step)
                    rotate_checkpoints(args, checkpoint_prefix="checkpoint")
                    self.save(current_file_name, args, optimizer, scheduler)
                    self._eval_loss(evaluation_loss, eval_dataloader, tb_writer, global_step)
                    if self._eval_during_training_custom(evaluator, args.output_dir, tb_writer, epoch, training_steps, global_step):
                        self._save_the_best_checkpoint(args, global_step, optimizer, scheduler)
                    loss_model.zero_grad()
                    loss_model.train()

            self._eval_loss(evaluation_loss, eval_dataloader, tb_writer, global_step)
            if self._eval_during_training_custom(evaluator, args.output_dir, tb_writer, epoch, -1, global_step):
                self._save_the_best_checkpoint(args, global_step, optimizer, scheduler)

        ## write results into pkl files.
        write_result(args, self.best_score)

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

    def _eval_during_training_custom(self, evaluator, output_dir, tb_writer, epoch, steps, global_step):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_dir, epoch=epoch, steps=steps)
            logger.info(f"The current score is {score:.3f} at {global_step} global step.")
            tb_writer.add_scalar("eval/score", score, global_step)
            if score > self.best_score:
                self.best_score = score
                return True
            return False

    def _save_the_best_checkpoint(self, args, global_step, optimizer, scheduler, best_file_name="checkpoint-best"):
        '''logging info and rm it first then save in case suboptimal.'''
        '''if ephemeral breaks in the middle saving the best checkpoint, please refer to the logging file.'''
        logger.info(f"The latest best checkpoint found at {global_step} global steps.")
        best_file_path = os.path.join(args.output_dir, best_file_name)
        if os.path.exists(best_file_path):
            shutil.rmtree(best_file_path)
            logger.info(f"Successfully removing the previous best checkpoint.")
        self.save(best_file_name, args, optimizer, scheduler)
