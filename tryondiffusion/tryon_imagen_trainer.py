import os
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from functools import partial
from math import ceil
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from ema_pytorch import EMA
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, random_split

from tryondiffusion import NullUnet
from tryondiffusion.common.python_helpers import (
    cast_tuple,
    cycle,
    default,
    exists,
    find_first,
    groupby_prefix_and_trim,
    num_to_groups,
    url_to_bucket,
)
from tryondiffusion.common.torch_helpers import cast_torch_tensor, eval_decorator, split


def split_args_and_kwargs(*args, split_size=None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [
        (
            split(arg, split_size=split_size)
            if exists(arg) and isinstance(arg, (torch.Tensor, Iterable))
            else ((arg,) * num_chunks)
        )
        for arg in all_args
    ]
    chunk_sizes = num_to_groups(batch_size, split_size)

    for chunk_size, *chunked_all_args in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = (
            chunked_all_args[:split_kwargs_index],
            chunked_all_args[split_kwargs_index:],
        )
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)


# imagen trainer


def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():
        if name not in state_dict_target:
            print(f"Layer {name} not found in target model, skipping...")
            continue

        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
            # print(f"Successfully loaded {name} with size {param.size()}.")
        else:
            print(f"Layer {name}({param.size()}) different than target: {state_dict_target[name].size()}.")

    return state_dict_target


class TryOnImagenTrainer(nn.Module):
    locked = False

    def __init__(
        self,
        imagen=None,
        use_ema: bool = False,
        lr=1e-4,
        eps=1e-8,
        betas=(0.9, 0.99),
        max_grad_norm=None,
        warmup_steps=None,
        cosine_decay_max_steps=None,
        only_train_unet_number=None,
        fp16=False,
        precision=None,
        split_batches=False,
        verbose=True,
        split_valid_fraction=0.025,
        split_valid_from_train=True,
        split_random_seed=42,
        init_checkpoint_path=None,
        checkpoint_path=None,
        checkpoint_every=None,
        checkpoint_fs=None,
        fs_kwargs: dict = None,
        max_checkpoints_keep=3,
        project_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # save config
        init_args = locals()
        init_args.pop("self")
        init_args.update(kwargs)
        self.config = init_args

        self.verbose = verbose

        assert (
            not TryOnImagenTrainer.locked
        ), "ImagenTrainer can only be initialized once per process - for the sake of distributed training, you will now have to create a separate script to train each unet (or a script that accepts unet number as an argument)"

        # determine filesystem, using fsspec, for saving to local filesystem or cloud

        self.fs = checkpoint_fs

        if not exists(self.fs):
            fs_kwargs = default(fs_kwargs, {})
            self.fs, _ = url_to_fs(default(checkpoint_path, "./"), **fs_kwargs)

        ema_kwargs, kwargs = groupby_prefix_and_trim("ema_", kwargs)

        # create accelerator instance
        accelerate_kwargs, kwargs = groupby_prefix_and_trim("accelerate_", kwargs)

        assert not (
            fp16 and exists(precision)
        ), 'either set fp16 = True or forward the precision ("fp16", "bf16") to Accelerator'
        accelerator_mixed_precision = default(precision, "fp16" if fp16 else "no")

        self.accelerator = Accelerator(
            **{
                "split_batches": split_batches,
                "mixed_precision": accelerator_mixed_precision,
                "kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=True)],
                **accelerate_kwargs,
            },
        )
        print(self.accelerator.state)
        if self.accelerator.is_local_main_process and exists(project_name):
            self.accelerator.init_trackers(project_name, config=self.config)

        TryOnImagenTrainer.locked = self.is_distributed

        # cast data to fp16 at training time if needed

        self.cast_half_at_training = accelerator_mixed_precision == "fp16"

        # grad scaler must be managed outside of accelerator

        grad_scaler_enabled = fp16

        # imagen, unets, and ema unets

        self.imagen = imagen
        self.num_unets = len(self.imagen.unets)

        self.use_ema = use_ema and self.is_main
        self.ema_unets = nn.ModuleList([])

        # keep track of what unet is being trained on
        # only going to allow 1 unet training at a time

        self.ema_unet_being_trained_index = -1  # keeps track of which ema unet is being trained on

        # data related functions

        self.train_dl_iter = None
        self.train_dl = None

        self.valid_dl_iter = None
        self.valid_dl = None

        # auto splitting validation from training, if dataset is passed in

        self.split_valid_from_train = split_valid_from_train

        assert 0 <= split_valid_fraction <= 1, "split valid fraction must be between 0 and 1"
        self.split_valid_fraction = split_valid_fraction
        self.split_random_seed = split_random_seed

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, eps, warmup_steps, cosine_decay_max_steps = map(
            partial(cast_tuple, length=self.num_unets), (lr, eps, warmup_steps, cosine_decay_max_steps)
        )

        for ind, (unet, unet_lr, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps) in enumerate(
            zip(self.imagen.unets, lr, eps, warmup_steps, cosine_decay_max_steps)
        ):
            optimizer = Adam(unet.parameters(), lr=unet_lr, eps=unet_eps, betas=betas)

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled=grad_scaler_enabled)

            scheduler = None

            if exists(unet_cosine_decay_max_steps):
                scheduler = CosineAnnealingLR(optimizer, T_max=unet_cosine_decay_max_steps)

            if exists(unet_warmup_steps):
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=unet_warmup_steps)

            # set on object

            setattr(self, f"optim{ind}", optimizer)  # cannot use pytorch ModuleList for some reason with optimizers
            setattr(self, f"scaler{ind}", scaler)
            setattr(self, f"scheduler{ind}", scheduler)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        # step tracker

        self.register_buffer("steps", torch.tensor([0] * self.num_unets))

        # automatic set devices based on what accelerator decided

        self.imagen.to(self.device)
        self.to(self.device)

        # checkpointing

        assert not (exists(checkpoint_path) ^ exists(checkpoint_every))
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints_keep = max_checkpoints_keep

        self.can_checkpoint = self.is_local_main if isinstance(checkpoint_fs, LocalFileSystem) else self.is_main

        if exists(checkpoint_path) and self.can_checkpoint:
            bucket = url_to_bucket(checkpoint_path)

            if not self.fs.exists(bucket):
                self.fs.mkdir(bucket)

            self.load_from_checkpoint_folder(init_checkpoint_path=init_checkpoint_path)

        # only allowing training for unet

        self.only_train_unet_number = only_train_unet_number
        self.prepared = False

    def prepare(self):
        assert not self.prepared, "The trainer is already prepared"
        self.validate_and_set_unet_being_trained(self.only_train_unet_number)
        self.prepared = True

    # computed values

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)

    # optimizer helper functions

    def get_lr(self, unet_number):
        self.validate_unet_number(unet_number)
        unet_index = unet_number - 1

        optim = getattr(self, f"optim{unet_index}")

        return optim.param_groups[0]["lr"]

    # function for allowing only one unet from being trained at a time

    def validate_and_set_unet_being_trained(self, unet_number=None):
        if exists(unet_number):
            self.validate_unet_number(unet_number)

        assert (
            not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number
        ), "you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet"

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)

    def wrap_unet(self, unet_number):
        if hasattr(self, "one_unet_wrapped"):
            return

        self.print(f"wrapping unet {unet_number} and preparing with accelerator")
        unet = self.imagen.get_unet(unet_number)
        unet_index = unet_number - 1

        optimizer = getattr(self, f"optim{unet_index}")
        scheduler = getattr(self, f"scheduler{unet_index}")

        if self.train_dl:
            self.unet_being_trained, self.train_dl, optimizer = self.accelerator.prepare(unet, self.train_dl, optimizer)
        else:
            self.unet_being_trained, optimizer = self.accelerator.prepare(unet, optimizer)

        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f"optim{unet_index}", optimizer)
        setattr(self, f"scheduler{unet_index}", scheduler)

        self.one_unet_wrapped = True

    # hacking accelerator due to not having separate gradscaler per optimizer

    def set_accelerator_scaler(self, unet_number):
        def patch_optimizer_step(accelerated_optimizer, method):
            def patched_step(*args, **kwargs):
                accelerated_optimizer._accelerate_step_called = True
                return method(*args, **kwargs)

            return patched_step

        unet_number = self.validate_unet_number(unet_number)
        scaler = getattr(self, f"scaler{unet_number - 1}")

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler
            optimizer._accelerate_step_called = False
            optimizer._optimizer_original_step_method = optimizer.optimizer.step
            optimizer._optimizer_patched_step_method = patch_optimizer_step(optimizer, optimizer.optimizer.step)

    # helper print

    def print(self, msg):
        if not self.is_main:
            return

        if not self.verbose:
            return

        return self.accelerator.print(msg)

    # validating the unet number

    def validate_unet_number(self, unet_number=None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert 0 < unet_number <= self.num_unets, f"unet number should be in between 1 and {self.num_unets}"
        return unet_number

    # number of training steps taken

    def num_steps_taken(self, unet_number=None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        return self.steps[unet_number - 1].item()

    def print_untrained_unets(self):
        print_final_error = False

        for ind, (steps, unet) in enumerate(zip(self.steps.tolist(), self.imagen.unets)):
            if steps > 0 or isinstance(unet, NullUnet):
                continue

            self.print(f"unet {ind + 1} has not been trained")
            print_final_error = True

        if print_final_error:
            self.print(
                "when sampling, you can pass stop_at_unet_number to stop early in the cascade, so it does not try to generate with untrained unets"
            )

    # data related functions

    def add_train_dataloader(self, dl=None):
        if not exists(dl):
            return

        assert not exists(self.train_dl), "training dataloader was already added"
        assert not self.prepared, "You need to add the dataset before preperation"
        self.train_dl = dl

    def add_valid_dataloader(self, dl):
        if not exists(dl):
            return

        assert not exists(self.valid_dl), "validation dataloader was already added"
        assert not self.prepared, "You need to add the dataset before preperation"
        self.valid_dl = dl

    def add_train_dataset(self, ds=None, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.train_dl), "training dataloader was already added"

        valid_ds = None
        if self.split_valid_from_train:
            train_size = int((1 - self.split_valid_fraction) * len(ds))
            valid_size = len(ds) - train_size

            ds, valid_ds = random_split(
                ds, [train_size, valid_size], generator=torch.Generator().manual_seed(self.split_random_seed)
            )
            self.print(
                f"training with dataset of {len(ds)} samples and validating with randomly splitted {len(valid_ds)} samples"
            )

        dl = DataLoader(ds, batch_size=batch_size, **dl_kwargs)
        self.add_train_dataloader(dl)

        if not self.split_valid_from_train:
            return

        self.add_valid_dataset(valid_ds, batch_size=batch_size, **dl_kwargs)

    def add_valid_dataset(self, ds, *, batch_size, **dl_kwargs):
        if not exists(ds):
            return

        assert not exists(self.valid_dl), "validation dataloader was already added"

        dl = DataLoader(ds, batch_size=batch_size, **dl_kwargs)
        self.add_valid_dataloader(dl)

    def create_train_iter(self):
        assert exists(self.train_dl), "training dataloader has not been registered with the trainer yet"

        if exists(self.train_dl_iter):
            return

        self.train_dl_iter = cycle(self.train_dl)

    def create_valid_iter(self):
        assert exists(self.valid_dl), "validation dataloader has not been registered with the trainer yet"

        if exists(self.valid_dl_iter):
            return

        self.valid_dl_iter = cycle(self.valid_dl)

    def train_step(self, *, unet_number=None, **kwargs):
        if not self.prepared:
            self.prepare()
        self.create_train_iter()

        kwargs = {"unet_number": unet_number, **kwargs}
        loss = self.step_with_dl_iter(self.train_dl_iter, **kwargs)
        self.update(unet_number=unet_number)
        return loss

    @torch.no_grad()
    @eval_decorator
    def valid_step(self, **kwargs):
        if not self.prepared:
            self.prepare()
        self.create_valid_iter()
        context = self.use_ema_unets if kwargs.pop("use_ema_unets", False) else nullcontext
        with context():
            loss = self.step_with_dl_iter(self.valid_dl_iter, **kwargs)
        return loss

    def step_with_dl_iter(self, dl_iter, **kwargs):
        sample = next(dl_iter)
        loss = self.forward(**{**kwargs, **sample})
        return loss

    # checkpointing functions

    @property
    def all_checkpoints_sorted(self):
        glob_pattern = os.path.join(self.checkpoint_path, "*.pt")
        checkpoints = self.fs.glob(glob_pattern)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(str(x).split(".")[-2]), reverse=True)
        return sorted_checkpoints

    def load_from_checkpoint_folder(self, last_total_steps=-1, init_checkpoint_path=None):
        if exists(init_checkpoint_path):
            self.load(init_checkpoint_path)
            return

        if last_total_steps != -1:
            filepath = os.path.join(self.checkpoint_path, f"checkpoint.{last_total_steps}.pt")
            self.load(filepath)
            return

        sorted_checkpoints = self.all_checkpoints_sorted

        if len(sorted_checkpoints) == 0:
            self.print(f"no checkpoints found to load from at {self.checkpoint_path}")
            return

        last_checkpoint = sorted_checkpoints[0]
        self.load(last_checkpoint)

    def save_to_checkpoint_folder(self):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        total_steps = int(self.steps.sum().item())
        filepath = os.path.join(self.checkpoint_path, f"checkpoint.{total_steps}.pt")

        self.save(filepath)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted
        checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep :]

        for checkpoint in checkpoints_to_discard:
            self.fs.rm(checkpoint)

    # saving and loading functions

    def save(self, path, overwrite=True, without_optim_and_sched=False, **kwargs):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        fs = self.fs

        assert not (fs.exists(path) and not overwrite)

        self.reset_ema_unets_all_one_device()

        save_obj = dict(model=self.imagen.state_dict(), steps=self.steps.cpu(), **kwargs)

        save_optim_and_sched_iter = range(0, self.num_unets) if not without_optim_and_sched else tuple()

        for ind in save_optim_and_sched_iter:
            scaler_key = f"scaler{ind}"
            optimizer_key = f"optim{ind}"
            scheduler_key = f"scheduler{ind}"

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)

            if exists(scheduler):
                save_obj = {**save_obj, scheduler_key: scheduler.state_dict()}

            save_obj = {**save_obj, scaler_key: scaler.state_dict(), optimizer_key: optimizer.state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, "ema": self.ema_unets.state_dict()}

        # save to path

        with fs.open(path, "wb") as f:
            torch.save(save_obj, f)

        self.print(f"checkpoint saved to {path}")

    def load(self, path, only_model=False, strict=True, noop_if_not_exist=False):
        fs = self.fs

        if noop_if_not_exist and not fs.exists(path):
            self.print(f"trainer checkpoint not found at {str(path)}")
            return

        assert fs.exists(path), f"{path} does not exist"

        self.reset_ema_unets_all_one_device()

        # to avoid extra GPU memory usage in main process when using Accelerate

        with fs.open(path) as f:
            loaded_obj = torch.load(f, map_location="cpu")

        try:
            self.print(f"Loading model checkpoint from {path}")
            self.imagen.load_state_dict(loaded_obj["model"], strict=strict)
        except RuntimeError:
            self.print("Failed loading state dict. Trying partial load")
            self.imagen.load_state_dict(restore_parts(self.imagen.state_dict(), loaded_obj["model"]))

        if only_model:
            return loaded_obj

        self.steps.copy_(loaded_obj["steps"])

        for ind in range(0, self.num_unets):
            scaler_key = f"scaler{ind}"
            optimizer_key = f"optim{ind}"
            scheduler_key = f"scheduler{ind}"

            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)

            if exists(scheduler) and scheduler_key in loaded_obj:
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(optimizer):
                try:
                    self.print("Loading optimizer and scaler...")
                    optimizer.load_state_dict(loaded_obj[optimizer_key])
                    scaler.load_state_dict(loaded_obj[scaler_key])
                except:
                    self.print(
                        "could not load optimizer and scaler, possibly because you have turned on mixed precision training since the last run. resuming with new optimizer and scalers"
                    )
        if self.use_ema:
            assert "ema" in loaded_obj
            try:
                self.ema_unets.load_state_dict(loaded_obj["ema"], strict=strict)
            except RuntimeError:
                print("Failed loading state dict. Trying partial load")
                self.ema_unets.load_state_dict(restore_parts(self.ema_unets.state_dict(), loaded_obj["ema"]))

        self.print(f"checkpoint loaded from {path}")
        return loaded_obj

    # managing ema unets and their devices

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def get_ema_unet(self, unet_number=None):
        if not self.use_ema:
            return

        unet_number = self.validate_unet_number(unet_number)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.ema_unets]
            delattr(self, "ema_unets")
            self.ema_unets = unets_list

        if index != self.ema_unet_being_trained_index:
            for unet_index, unet in enumerate(self.ema_unets):
                unet.to(self.device if unet_index == index else "cpu")

        self.ema_unet_being_trained_index = index
        return self.ema_unets[index]

    def reset_ema_unets_all_one_device(self, device=None):
        if not self.use_ema:
            return

        device = default(device, self.device)
        self.ema_unets = nn.ModuleList([*self.ema_unets])
        self.ema_unets.to(device)

        self.ema_unet_being_trained_index = -1

    @torch.no_grad()
    @contextmanager
    def use_ema_unets(self):
        if not self.use_ema:
            output = yield
            return output

        self.reset_ema_unets_all_one_device()
        self.imagen.reset_unets_all_one_device()

        self.unets.eval()

        trainable_unets = self.imagen.unets
        self.imagen.unets = self.unets  # swap in exponential moving averaged unets for sampling

        output = yield

        self.imagen.unets = trainable_unets  # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    def print_unet_devices(self):
        self.print("unet devices:")
        for i, unet in enumerate(self.imagen.unets):
            device = next(unet.parameters()).device
            self.print(f"\tunet {i}: {device}")

        if not self.use_ema:
            return

        self.print("\nema unet devices:")
        for i, ema_unet in enumerate(self.ema_unets):
            device = next(ema_unet.parameters()).device
            self.print(f"\tema unet {i}: {device}")

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_ema_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # forwarding functions and gradient step updates

    def update(self, unet_number=None):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        index = unet_number - 1
        unet = self.unet_being_trained

        optimizer = getattr(self, f"optim{index}")
        scaler = getattr(self, f"scaler{index}")
        scheduler = getattr(self, f"scheduler{index}")

        # set the grad scaler on the accelerator, since we are managing one per u-net

        if self.accelerator.sync_gradients and exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.get_ema_unet(unet_number)
            ema_unet.update()

        # scheduler, if needed

        if exists(scheduler) and not self.accelerator.optimizer_step_was_skipped:  # recommended in the docs
            scheduler.step()

        self.steps += F.one_hot(torch.tensor(unet_number - 1, device=self.steps.device), num_classes=len(self.steps))

        if not exists(self.checkpoint_path):
            return

        total_steps = int(self.steps.sum().item())

        if total_steps % self.checkpoint_every:
            return

        self.save_to_checkpoint_folder()

    @torch.no_grad()
    @cast_torch_tensor
    def sample(self, *args, **kwargs):
        context = nullcontext if kwargs.pop("use_non_ema", False) else self.use_ema_unets

        self.print_untrained_unets()
        if not self.is_main:
            kwargs["use_tqdm"] = False

        with context():
            output = self.imagen.sample(*args, device=self.device, **kwargs)

        return output

    @partial(cast_torch_tensor, cast_fp16=True)
    def forward(self, *args, unet_number=None, **kwargs):
        unet_number = self.validate_unet_number(unet_number)
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        assert (
            not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number
        ), f"you can only train unet #{self.only_train_unet_number}"

        with self.accelerator.accumulate(self.unet_being_trained):
            with self.accelerator.autocast():
                loss = self.imagen(*args, unet=self.unet_being_trained, unet_number=unet_number, **kwargs)

            if self.training:
                self.accelerator.backward(loss)

        return loss


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
