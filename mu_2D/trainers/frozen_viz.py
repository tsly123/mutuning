import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from tabulate import tabulate

from trainers.vision_benchmark.evaluation import construct_dataloader, construct_multitask_dataset

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

import numpy as np
# torch.set_printoptions(precision=10)
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


from torch.nn import Dropout
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair


class ImageEncoder(nn.Module):
    def __init__(self, clip_model, classnames):
        super().__init__()
        # HACK: Assume all is vision transformer
        self.visual = clip_model.visual
        embed_dim = clip_model.text_projection.shape[1]
        # print('embed_dim', embed_dim)

        # Classifier Head
        self.head = nn.Sequential(torch.nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6),
                                  nn.Linear(embed_dim, len(classnames)))

    def forward(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                             device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        B = x.shape[0]

        x = x.permute(1, 0, 2)  # NLD -> LND

        for layer_idx in range(self.visual.transformer.layers):
            layer = self.visual.transformer.resblocks[layer_idx]
            x = layer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        # print('img_enc', x.shape)
        # x = x / x.norm(dim=-1, keepdim=True)
        # x = self.fc_norm(x)
        x = self.head(x)

        return x

    def get_intermediate_layers(self, x: torch.Tensor):
        weights_all_blocks = []
        weights_attnout_blocks = []

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                             device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        B = x.shape[0]

        x = x.permute(1, 0, 2)  # NLD -> LND

        for layer_idx in range(self.visual.transformer.layers):
            layer = self.visual.transformer.resblocks[layer_idx]
            x, attnout_weight = layer(x, return_attn_weight=True)


            x_out = x.permute(1, 0, 2)  # LND -> NLD
            attnout_weight = attnout_weight.permute(1, 0, 2)

            weights_all_blocks.append(x_out)
            weights_attnout_blocks.append(attnout_weight)

        x = x.permute(1, 0, 2)  # LND -> NLD
        class_token = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            class_token = class_token @ self.visual.proj

        return class_token, weights_all_blocks, weights_attnout_blocks

    def get_selfattention(self, x: torch.Tensor):

        weights_all_blocks = []

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                             device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        B = x.shape[0]

        x = x.permute(1, 0, 2)  # NLD -> LND

        for layer_idx in range(self.visual.transformer.layers):
            layer = self.visual.transformer.resblocks[layer_idx]

            x, weight = layer(x, return_attention=True)
            # print('check attn weight', weight.shape)

            weights_all_blocks.append(weight)

        return weights_all_blocks
    
# text branch is eval
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, dm=None):
        super().__init__()
        # self.prompt_learner = MultitaskVLPromptLearner(cfg, classnames, clip_model)
        # self.classnames = classnames
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model, classnames)
        self.image_encoder.train()
        # self.text_encoder = TextEncoder(clip_model, cfg)
        # self.text_encoder.eval()
        # self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # self.clip_model = clip_model
        # self.max_length = clip_model.context_length
        # self.token_embedding = clip_model.token_embedding

        self.multi_task_label_pertask = cfg.DATASET.MULTITASK_LABEL_PERTASK
        if self.multi_task_label_pertask:
            self.class_index_pertask_start = torch.arange(dm._num_classes)
            self.class_index_pertask_end = torch.arange(dm._num_classes)
            start_index = 0

            for class_index, task in enumerate(dm._task_names):
                class_num = len(dm._labelmap[task])
                self.class_index_pertask_start[class_index] = start_index
                start_index += class_num
                self.class_index_pertask_end[class_index] = start_index
            self.index = torch.arange(dm._num_classes).unsqueeze(0)

    def forward(self, image, task=None):
        # coop_emb = self.prompt_learner.forward_mvlpt_proj(self.dtype)
        # print('coop_emb', coop_emb)
        logits = self.image_encoder(image.type(self.dtype))


        if self.multi_task_label_pertask:
            # Here we perform prompt selection
            domain_start_indexs = self.class_index_pertask_start[task].unsqueeze(-1)
            domain_end_indexs = self.class_index_pertask_end[task].unsqueeze(-1)
            # print(domain_start_indexs.shape, domain_end_indexs.shape, logits.shape)
            select_index = self.index.repeat(logits.shape[0], 1)
            select_index = (select_index >= domain_start_indexs).float() * (select_index < domain_end_indexs).float()
            # exit()
            logits = logits * select_index.to(logits.device)
        # print('logits', logits.shape)
        return logits


class MVLPTCOOPDataManager(DataManager):

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        self._id2task = {}
        self._task_class_idx = {}

        for domain, dataset_name in enumerate(self._task_names):
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)
            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}
            for key, value in dataset._lab2cname.items():
                new_lab2cname_dict[key + label_offset] = value
            self.lab2cname_list.update(new_lab2cname_dict)
            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain

            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain
                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain

            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._val = self.dataset.val + dataset.val
                self.dataset._test = self.dataset.test + dataset.test

            print(dataset._train_u is None, dataset._val is None)
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = (label_offset, label_offset + dataset._num_classes)
            label_offset += dataset._num_classes

        dataset = self.dataset
        dataset._classnames = self.classnames_list
        dataset._lab2cname = self.lab2cname_list
        dataset._num_classes = sum(self.num_classes_list)
        print(self.num_classes_list, len(dataset._classnames), dataset._lab2cname, dataset._num_classes)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )
        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

        # print('CHECK DATASET LEN')
        # print('TRAIN', len(dataset.train_x))
        # print('VAL', len(dataset.val))
        # print('TEST', len(dataset.test))
        # if dataset.train_u:
        #     print('TRAIN U ', len(dataset.train_u))


from trainers.vision_benchmark.datasets import class_map_metric, get_metric
import random


class MVLPTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, class_map, train_dataset = construct_dataloader(cfg)

        self._metric = get_metric(class_map_metric[cfg.DATASET.DATASET])
        self._metric_name = class_map_metric[cfg.DATASET.DATASET]

        # Attributes
        self._num_classes = len(class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in enumerate(class_map):
            if isinstance(value, list):
                # value = random.choice(value)
                value = value[0]
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass
            # self.show_dataset_summary(cfg)


class MVLPTMTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, train_dataset, test_dataloader_by_task = construct_multitask_dataset(
            cfg)

        self._labelmap = train_dataset.labelmap
        self._task_names = train_dataset._task_names
        self._task2id = {v: k for k, v in enumerate(self._task_names)}
        self._id2task = {k: v for k, v in enumerate(self._task_names)}
        self._metric = {task: get_metric(class_map_metric[task]) for task in self._task_names}
        self._metric_name = {task: class_map_metric[task] for task in self._task_names}

        class_idx = 0
        self._task_class_idx = {}
        for task in self._task_names:
            class_num = len(self._labelmap[task])
            self._task_class_idx[task] = (class_idx, class_idx + class_num)
            class_idx += class_num

        from trainers.vision_benchmark.datasets import class_map

        print(self._task_names)
        print(self._labelmap)
        print(class_map.keys())

        mt_class_map = dict()
        for task in self._labelmap:
            for label_idx, label in enumerate(class_map[task]):
                cnt = train_dataset._get_cid(label_idx, task)
                mt_class_map[cnt] = label

        print(mt_class_map)
        # Attributes
        self._num_classes = len(mt_class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}

        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in mt_class_map.items():
            if isinstance(value, list):
                value = value[0]  # random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass


@TRAINER_REGISTRY.register()
class Baseline_trainer(TrainerX):
    """Context Optimization (MVLPT).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Baseline.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        if self.cfg.DATASET.COOP:
            classnames = self.dm.dataset.classnames
        else:
            classnames = self.dm.lab2cname.values()

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # clip_model.train() for visual branch
        clip_model.train()

        if cfg.TRAINER.Baseline.PREC == "fp32" or cfg.TRAINER.Baseline.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, dm=self.dm)
        # print('self.model', self.model)

        if cfg.TRAINER.Baseline.PREC == "fp16":
            self.model.image_encoder.head[-1].weight.data = self.model.image_encoder.head[-1].weight.data.half()
            if self.model.image_encoder.head[-1].bias is not None:
                self.model.image_encoder.head[-1].bias.data = self.model.image_encoder.head[-1].bias.data.half()


        print("Turning gradients in the text encoder")
        for name, param in self.model.named_parameters():
            # print(name)
            # if "image_encoder" not in name:
            #     param.requires_grad_(False)
            # else:
            #     param.requires_grad_(True)
                # print(name)
            param.requires_grad_(False)


        print(
            f"Tunable Param: {sum([p.numel() for p in self.model.parameters() if p.requires_grad]) / 10 ** 6}M, Original CLIP {sum([p.numel() for p in self.model.parameters() if not p.requires_grad]) / 10 ** 6}M")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        param_groups = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_groups.append({'params': param})
                print(name, param.shape)

        # NOTE: optimize entire model
        self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM, param_groups=param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("baseline_model", self.model.image_encoder, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.Baseline.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        self.multi_task = self.cfg.DATASET.MULTITASK
        self.multi_task_label_pertask = self.cfg.DATASET.MULTITASK_LABEL_PERTASK

        if self.cfg.DATASET.COOP:
            dm = MVLPTCOOPDataManager(self.cfg)
        elif self.cfg.DATASET.MULTITASK:
            dm = MVLPTMTDataManager(self.cfg)
        else:
            dm = MVLPTDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def forward_backward(self, batch):
        image, label, tasks_ = self.parse_batch_train(batch)

        # HACK: for multi-label classification, either works
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = label.float()
            label /= label.sum(dim=-1, keepdim=True)

        prec = self.cfg.TRAINER.Baseline.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, task=tasks_)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, task=tasks_)
            # print(label.shape, output.shape, label.dtype, output.dtype, tasks_, label.sum(dim=-1))

            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        # HACK: During training, we hack the eval of multi-label by selecting only one class
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = torch.argmax(label, dim=1)

        # result = self.dm._metric(label.squeeze().cpu().detach().numpy(), output.cpu().detach().numpy())

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            # "acc": result,
        }
        if tasks_ is not None:
            loss_summary.update({"num_tasks": len(set(tasks_.tolist()))})

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        # print(label.shape, 'label', input.shape, 'input')
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def parse_batch_test(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def model_inference(self, input, task=None):
        return self.model(input, task=task)

    @torch.no_grad()
    def test(self, split=None):
        from tqdm import tqdm
        import copy
        import numpy as np
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        self.evaluator_task = dict()

        self.elevator_evaluator = {'y_pred': [], 'y_true': []}

        if self.multi_task:
            if self.cfg.DATASET.COOP:
                self.evaluator_task = {task: copy.deepcopy(self.evaluator) for task in self.dm._task_names}
            else:
                self.evaluator_task = {task: copy.deepcopy(self.elevator_evaluator) for task in self.dm._task_names}

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, tasks_ = self.parse_batch_test(batch)
            output = self.model_inference(input, task=tasks_)
            # HACK: make everything one-hot vector label!
            if self.cfg.DATASET.COOP:
                self.evaluator.process(output, label)

            else:
                self.elevator_evaluator['y_pred'].append(output.cpu().detach().numpy())
                self.elevator_evaluator['y_true'].append(label.cpu().detach().numpy())

            if tasks_ is not None:
                for out, lab, task in zip(output, label, tasks_):
                    task = self.dm._id2task[task.item()]

                    if self.cfg.DATASET.COOP:
                        class_start, class_end = self.dm._task_class_idx[task]
                        # Evaluate on the task-specific class
                        out = out[class_start:class_end]
                        lab -= class_start
                        self.evaluator_task[task].process(out.unsqueeze(0), lab.unsqueeze(0))
                    else:
                        self.evaluator_task[task]['y_pred'].append([out.cpu().detach().numpy()])
                        self.evaluator_task[task]['y_true'].append([lab.cpu().detach().numpy()])

        results_overall = {}
        for task in self.evaluator_task:
            print(f"evaluate on the *{task}* !")
            if self.cfg.DATASET.COOP:
                results = self.evaluator_task[task].evaluate()
                results_overall[task] = results['accuracy']
            else:
                y_true = np.concatenate(self.evaluator_task[task]['y_true'], axis=0)
                y_pred = np.concatenate(self.evaluator_task[task]['y_pred'], axis=0)
                class_start, class_end = self.dm._task_class_idx[task]
                y_true = y_true[:, class_start:class_end]
                y_pred = y_pred[:, class_start:class_end]

                if self.dm._metric_name[task] == 'accuracy':
                    y_true = np.argmax(y_true, axis=-1)
                metric_result = self.dm._metric[task](y_true, y_pred)
                results = {self.dm._metric_name[task]: metric_result}
                results_overall[task] = metric_result
            print('results', results)
            for k, v in results.items():
                tag = f"{split}/{task}/{k}"
                self.write_scalar(tag, v, self.epoch)

        print(f"Overall evaluation !")
        if self.multi_task:
            multi_task_evalkey = self.cfg.DATASET.MULTITASK_EVALKEY
            if multi_task_evalkey == 'average':
                results = {'average': sum([v for k, v in results_overall.items()]) / len(results_overall)}
            else:
                assert multi_task_evalkey in results_overall
                results = {multi_task_evalkey: results_overall[multi_task_evalkey]}
                print(f"select {multi_task_evalkey} as the evaluation key")
        else:
            if not self.cfg.DATASET.COOP:
                y_true = np.concatenate(self.elevator_evaluator['y_true'], axis=0)
                y_pred = np.concatenate(self.elevator_evaluator['y_pred'], axis=0)
                results = {self.dm._metric_name: self.dm._metric(y_true, y_pred)}
            else:
                results = self.evaluator.evaluate()

        # compute AUC
        auc, sen, spec = self.get_auc(results)
        print(f"* AUC: {auc:.2f}")
        print(f"* Sensitivity: {sen:.2f}")
        print(f"* Specificity: {spec:.2f}")
        del results["_y_true"]
        del results["_y_pred"]
        results["AUC"] = auc
        results["Sensitivity"] = sen
        results["Specificity"] = spec

        print('results', results)
        for k, v in results.items():
            tag = f"/{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results["accuracy"], results["AUC"]

    def get_auc(self, results):

        _y_true = results["_y_true"]
        _y_pred = results["_y_pred"]
        # print('_y_true', _y_true)
        # print('_y_pred', _y_pred)

        true = np.array(_y_true)
        pred = np.array(_y_pred)

        if true.max() == 1:  # binary-class
            # print('true', true)
            # print('pred', pred)
            auc = roc_auc_score(true, pred)
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            return auc, sensitivity, specificity

        else:  # multi-classes
            auc = 0
            sensitivity = 0
            specificity = 0
            for i in range(true.max() + 1):
                y_true_binary = np.where(true == i, 1, 0)
                y_score_binary = np.where(pred == i, 1, 0)
                temp_auc = roc_auc_score(y_true_binary, y_score_binary)
                auc += temp_auc
                tn, fp, fn, tp = confusion_matrix(y_true_binary, y_score_binary).ravel()
                sen = tp / (tp + fn)
                spec = tn / (tn + fp)
                print(f"Class {i}: AUC {temp_auc:.2f}, sensitivity {sen:.2f}, specificity {spec:.2f}")
                sensitivity += sen
                specificity += spec
            ret = auc / (true.max() + 1)
            ret_sen = sensitivity / (true.max() + 1)
            ret_spec = specificity / (true.max() + 1)
            return ret, ret_sen, ret_spec

    def load_model(self, directory, epoch=None, name_model=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
            
        if name_model is not None:
            model_file = name_model

        for name in names:
            print('load model name', name)
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # issue 1 bug fix for UPT key name mismatch
            # state_dict = {k.replace("upt_proj", "mvlpt_proj"): v for k, v in state_dict.items()}
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors

            # if "token_prefix" in state_dict:
            #     del state_dict["token_prefix"]
            #
            # if "token_suffix" in state_dict:
            #     del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def viz_prosess(self, img, attentions, threshold=None, patch_size=16, output_dir=None, img_id=0):
        # attn_list = []

        # attn_list.append(img.cpu().numpy())
        # print('viz_prosess 1', img.shape)

        # image size should divisible by patch_size
        w_featmap = self.cfg.INPUT.SIZE[0] // patch_size
        h_featmap = self.cfg.INPUT.SIZE[1] // patch_size

        attentions = attentions.unsqueeze(0)
        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
                0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0].cpu().numpy()

        # save attentions heatmaps
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                     os.path.join(output_dir, f"id{img_id}_img.png"))
        for j in range(nh):
            fname = os.path.join(output_dir, f"id{img_id}_attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png', cmap=plt.cm.jet)
            print(f"{fname} saved.")
            # attn_list.append(attentions[j])
            # print('viz_prosess 2', attentions[j].shape)

        attentions_mean = attentions.mean(0)
        fname = os.path.join(output_dir, f"id{img_id}_attn-mean.png")
        plt.imsave(fname=fname, arr=attentions_mean, format='png', cmap=plt.cm.jet)
        # print('viz_prosess 3', attentions_mean.shape)

        # attn_list.append(attentions_mean)
        # # attn_list = np.swapaxes(np.stack(attn_list), 0,2)
        # attn_list = np.concatenate(attn_list, 1)#.reshape(224*4, -1)
        # print('viz_prosess 4', attn_list.shape)
        # # torchvision.utils.save_image(torchvision.utils.make_grid(torch.from_numpy(attn_list).float().unsqueeze(1), normalize=True, scale_each=True, nrow=3),
        # #                              os.path.join(output_dir, f"id{img_id}_attn.png"))
        #
        # fname = os.path.join(output_dir, f"id{img_id}_attn.png")
        # plt.imsave(fname=fname, arr=attn_list, format='png')

        if threshold is not None:
            image = skimage.io.imread(os.path.join(output_dir, f"id{img_id}_img.png"))
            for j in range(nh):
                viz.display_instances(image, th_attn[j], fname=os.path.join(output_dir, f"id{img_id}_mask_th" + str(
                    threshold) + "_head" + str(j) + ".png"), blur=False)
            th_attn_mean = th_attn.mean(0)
            fname_th_attn = os.path.join(output_dir, f"id{img_id}_mask_th" + str(threshold) + "_mean" + ".png")
            viz.display_instances(image, th_attn_mean, fname=fname_th_attn)

    @torch.no_grad()
    def viz(self, split=None, name_model=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        # self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        # output_dir = f"{self.cfg.OUTPUT_DIR}/viz/{name_model}"
        output_dir = f"{self.cfg.OUTPUT_DIR}/viz_savefig"
        os.makedirs(output_dir, exist_ok=True)

        count = 0

        # DINO
        # for batch_idx, batch in enumerate(tqdm(data_loader)):
        #     img, _, _ = self.parse_batch_test(batch)
        #     # output = self.model_inference(input, task=tasks_)
        #     # HACK: make everything one-hot vector label!
        #     attn = self.model.get_last_selfattention(img)
        #     # print('viz 0', attn.shape)    # torch.Size([7488, 213, 213])
        #     slice_id = [0] + list(range(1+self.cfg.TRAINER.MVLPT.VPT.N_CTX, attn.shape[-1]))
        #     # print('viz slice_id', len(slice_id), slice_id)
        #     attn = attn[:, slice_id]#[slice_id]
        #     attn = attn[:, :, slice_id]  # [slice_id]
        #     # print('viz 1', attn.shape)
        #     attn = attn.view(-1, 12, len(slice_id), len(slice_id))
        #     # print('viz 2', attn.shape)
        #     for attn_i in range(attn.shape[0]):
        #
        #         # print('viz 2', attn[attn_i].shape)
        #         img_id = count * self.cfg.DATALOADER.TEST.BATCH_SIZE + attn_i
        #         self.viz_prosess(img[attn_i], attn[attn_i], patch_size=16, output_dir=output_dir, img_id=img_id)
        #
        #     count += 1
        #     break

        true_all, pred_all = self.test(get_output=True)
        if not os.path.isfile(f"{output_dir}-true.txt"):
            np.savetxt(f"{output_dir}-true.txt", true_all)
            np.savetxt(f"{output_dir}-pred.txt", pred_all)

        # jet
        for batch_idx, batch in enumerate(data_loader):
            img, label, tasks_ = self.parse_batch_test(batch)
            print('viz img', img.shape)
            # output = self.model_inference(input, task=tasks_)
            # HACK: make everything one-hot vector label!
            # output = self.model_inference(img, task=tasks_)

            # continue

            attn = self.model.get_selfattention(img)
            print('viz 0', attn.shape)  # [12, 7488, 213, 213]

            # slice_id = [0] + list(range(1+self.cfg.TRAINER.MVLPT.VPT.N_CTX, attn.shape[-1]))
            # attn = attn[:, :, slice_id]
            # attn = attn[:, :, :, slice_id]  # [slice_id]
            # attn = attn.view(12, -1,  len(slice_id), len(slice_id))

            true = true_all[batch_idx * self.cfg.DATALOADER.TEST.BATCH_SIZE: (
                                                                                         batch_idx + 1) * self.cfg.DATALOADER.TEST.BATCH_SIZE]
            pred = pred_all[batch_idx * self.cfg.DATALOADER.TEST.BATCH_SIZE: (
                                                                                         batch_idx + 1) * self.cfg.DATALOADER.TEST.BATCH_SIZE]

            assert true.shape[0] == attn.shape[1]
            assert pred.shape[0] == attn.shape[1]

            for attn_i in range(attn.shape[1]):
                # print('viz 2', attn[attn_i].shape)
                img_id = count * self.cfg.DATALOADER.TEST.BATCH_SIZE + attn_i

                img_pp = img[attn_i]
                # print('viz 3', img_pp.shape)
                img_attn_weights = attn[:, attn_i, ...]

                fname = os.path.join(output_dir, f"id{img_id}_img.png")
                img_save = img_pp.clone()
                img_save -= img_save.min()
                img_save /= img_save.max()
                plt.imsave(fname=fname, arr=img_save.permute(1, 2, 0).cpu().numpy(), format='png')

                self.get_image_attn_mask(img_pp, img_attn_weights, img_id=img_id, output_dir=output_dir,
                                         output=(true[attn_i], pred[attn_i]))

            count += 1
            # break

        print('Done generating attention maps!!!')

    @torch.no_grad()
    def get_image_attn_mask(self, img, att_mat, token=0, img_id=0, output_dir=None, output=None):
        """
        https://github.com/ricardodeazambuja/CLIP/blob/attn_weights/notebooks/Interacting_with_CLIP.ipynb
        https://github.com/facebookresearch/mmf/issues/145
        image.shape => [img_dim, img_dim]
        grid => img_dim//patch_size
        token_seq_length => 1+grid**2, where the "1+" comes from the addition of CLS token
        att_mat.shape => [n_layers, n_heads, token_seq_length, token_seq_length]
        """
        # img_cpu = img.permute(1,2,0).cpu().numpy()
        # img_cpu = Image.fromarray((img_cpu * 1).astype(np.uint8))
        # img_cpu = Image.fromarray(img_cpu)
        # img_cpu = Image.fromarray((img.transpose(0, 2).transpose(1, 0).cpu().numpy() *255).astype(np.uint8))
        # print('get_image_attn_mask img_cpu', img_cpu.size)
        mul_img = img.transpose(0, 2).transpose(1, 0).cpu().numpy() * 255
        print('mul_img max', mul_img.max())
        mul_img -= mul_img.min()
        mul_img /= mul_img.max()
        # mul_img = (mul_img - np.min(mul_img)) / (np.max(mul_img) - np.min(mul_img))
        print('get_image_attn_mask mul_img', mul_img.shape)

        # Heavily based on:
        # from https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
        att_mat = att_mat.cpu()  # [n_layers, n_heads, token_seq_length, token_seq_length]

        # Average the attention weights across all heads.
        # att_mat = att_mat.mean(dim=1)  # [n_layers, token_seq_length, token_seq_length]
        # aug_att_mat = att_mat.float()  # torch.matmul won't work with fp16

        #
        # CLIP uses a series of ResidualAttentionBlocks that look like this:
        # input x => LN => ATTN => ADD x => LN => MLP => ADD x => output y
        #
        # But the input x is NOT a simple image at all. It is the result of
        # a conv2d with a kernel and stride the same size so it "chops" the image into
        # something like patches with a lot of layers (the token embedding dimension).
        # A class (CLS) token (learned and frozen for inference) is concatenated to the start and to the
        # whole "sequence" we add the positional encoder (again, learned and frozen for inference)
        #
        # Recursively multiply the weight matrices
        # joint_attentions = torch.zeros(aug_att_mat.size())
        # joint_attentions[0] = aug_att_mat[0]
        # for n in range(1, aug_att_mat.size(0)):
        #     tmp_mat = aug_att_mat[n] + joint_attentions[n - 1]
        #     tmp_mat /= tmp_mat.norm()
        #     # ignoring the effects of the MLP...
        #     joint_attentions[n] = torch.matmul(tmp_mat, joint_attentions[n - 1])

        #######################################
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        aug_att_mat = aug_att_mat.float()

        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        #######################################

        # Attention from the output token to the input space.
        # if avg:
        #     v = joint_attentions.mean(dim=0)
        # else:
        #     v = joint_attentions[layer]

        for layer_i in range(att_mat.shape[0]):
            if layer_i != 11:
                continue
            v = joint_attentions[layer_i]

            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            att = torch.arange(att_mat.size(1))
            att = att[att != att[token]]
            mask = v[token, att].reshape(grid_size,
                                         grid_size).numpy()  # token=0 is CLS (the only one used by CLIP at the end)
            print('mask', mask.shape)
            # mask = np.asarray(Image.fromarray(mask.numpy()).resize(img_cpu.size)).copy()
            mask = cv2.resize(mask / mask.max(), (mul_img.shape[0], mul_img.shape[1])).copy()  # * 255
            mask -= mask.min()
            mask /= mask.max()

            # mul_img2 = mul_img * 0.1
            # result = (mask[..., np.newaxis] * mul_img2)
            # result -= result.min()
            # result /= result.max()

            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255

            # heatmap = (1 - heatmap) * 0.5
            # heatmap = np.float32(np.uint8(255 * mask)) / 255

            result = heatmap * np.float32(mul_img)
            # result = cam / np.max(cam)

            # result -= result.min()
            # result /= result.max()

            print('get_image_attn_mask result', result.shape)

            fname = os.path.join(output_dir, f"id{img_id}_true{output[0]}_pred{output[1]}_attn-layer{layer_i}.png")
            plt.imsave(fname=fname, arr=result, format='png', cmap=plt.cm.jet)

            # fig, ax = plt.subplots(ncols=1, nrows=1)
            # ax.imshow(result, cmap=plt.cm.jet)
            # # save_img.set_cmap('jet')
            # plt.axis('off')
            # plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close()

        # cmap = plt.cm.get_cmap('jet')
        # cmap.set_bad(color="k", alpha=0.0)

        v = joint_attentions.mean(dim=0)
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        att = torch.arange(att_mat.size(1))
        att = att[att != att[token]]
        mask = v[token, att].reshape(grid_size,
                                     grid_size).numpy()  # token=0 is CLS (the only one used by CLIP at the end)
        mask = cv2.resize(mask / mask.max(), (mul_img.shape[0], mul_img.shape[1])).copy()  # * 255
        mask -= mask.min()
        mask /= mask.max()

        # mask = 1 - mask

        # mul_img = cv2.cvtColor(mul_img, cv2.COLOR_RGB2RGBA)
        # b, g, r, a = cv2.split(mul_img)  # get b, g, r
        # mul_img = cv2.merge([r, g, b, a])
        #
        # mask = mask[..., np.newaxis]
        # # print('mask', mask.shape)
        # mask = np.array(Image.fromarray(cmap(mask, bytes=True), 'RGBA'))
        # mask = mask.astype(mul_img.dtype)
        # # print('mask', mask.shape)
        # result = cv2.addWeighted(mul_img, 0.7, mask, 0.5, 0)
        # result = result.astype(mul_img.dtype)

        # mask = mask[..., np.newaxis]
        # mask = mask * 0.95 + 0.05
        # result = mask * mul_img + (1 - mask) * 255
        # result = result.astype(mul_img.dtype)

        # mul_img2 = mul_img * 0.1
        # result = (mask[..., np.newaxis] * mul_img2)#*255
        # result -= result.min()
        # result /= result.max()

        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # heatmap = np.float32(np.uint8(255 * mask)) / 255

        # heatmap = (1 - heatmap) * 0.5

        result = heatmap * np.float32(mul_img)
        # result = cam / np.max(cam)

        # result -= result.min()
        # result /= result.max()

        # mask_save = mask[..., np.newaxis]

        # cmap = plt.cm.jet
        # norm = plt.Normalize(vmin=result.min(), vmax=result.max())
        # result = cmap(norm(result))
        # print('result', result.shape)
        fname = os.path.join(output_dir, f"id{img_id}_true{output[0]}_pred{output[1]}_attn-mean.png")

        plt.imsave(fname=fname, arr=result, format='png', cmap=plt.cm.jet)

        # fig, ax = plt.subplots(ncols=1, nrows=1)
        # ax.imshow(result, cmap=plt.cm.jet)
        # # save_img.set_cmap('jet')
        # plt.axis('off')
        # plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0)
        # plt.close()

        mask_r = 1 - mask

        heatmap = cv2.applyColorMap(np.uint8(255 * mask_r), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        result = heatmap * np.float32(mul_img)

        # result -= result.min()
        # result /= result.max()

        fname = os.path.join(output_dir, f"id{img_id}_true{output[0]}_pred{output[1]}_attn-mean-r.png")

        plt.imsave(fname=fname, arr=result, format='png', cmap=plt.cm.jet)
        plt.close()