# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import argparse
import torch
import dnnlib
import numpy as np
import torch.multiprocessing as mp
from torch_utils import distributed as dist
from training import training_loop_ssdisk

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def FloatRange(min=None, min_open=False, max=None, max_open=False):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: 
         min - minimum acceptable argument
         max - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if min is not None:
            if min_open and f <= min:
                raise argparse.ArgumentTypeError("must be larger than" + str(min))
            elif not min_open and f < min:
                raise argparse.ArgumentTypeError("must be larger than" + str(min))
        if max is not None:
            if max_open and f >= max:
                raise argparse.ArgumentTypeError("must be smaller than" + str(max))
            elif not max_open and f > max:
                raise argparse.ArgumentTypeError("must be smaller than" + str(max))
        return f

    # Return function handle to checking function
    return float_range_checker


def IntRange(min=None, min_open=False, max=None, max_open=False):
    """Return function handle of an argument type function for
       ArgumentParser checking a int range: 
         min - minimum acceptable argument
         max - maximum acceptable argument"""

    # Define the function with default arguments
    def int_range_checker(arg):
        """New Type function for argparse - an int within predefined range."""

        try:
            f = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer number")
        if min is not None:
            if min_open and f <= min:
                raise argparse.ArgumentTypeError("must be larger than" + str(min))
            elif not min_open and f < min:
                raise argparse.ArgumentTypeError("must be larger than" + str(min))
        if max is not None:
            if max_open and f >= max:
                raise argparse.ArgumentTypeError("must be smaller than" + str(max))
            elif not max_open and f > max:
                raise argparse.ArgumentTypeError("must be smaller than" + str(max))
        return f

    # Return function handle to checking function
    return int_range_checker

#----------------------------------------------------------------------------


def train(rank, world_size, hyperparams):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """

    opts = dnnlib.EasyDict(hyperparams)

    print(f"Initializing DDP training rank {rank}.", flush=True)
    print(rank, hyperparams, flush=True)
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    if world_size == 1:
        os.environ['MASTER_PORT'] = str(np.random.randint(low=1024, high=16384))
    else:
        os.environ['MASTER_PORT'] = str(12355)

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Initialize config dict.
    c = dnnlib.EasyDict()
    
    config = {}
    config['image_size'] = 128
    config['prior_Rg'] = [0.5, 2]
    config['prior_Rin'] = [3, 12]
    config['prior_slope'] = [1, 5]
    config['prior_phi0'] = [0, 2*np.pi]
    config['prior_cosinc'] = [0, 1]

    c.dataset_kwargs = dnnlib.EasyDict(config)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'SSdisk-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop_ssdisk.training_loop_ssdisk(**c)

    torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Main options.
    parser.add_argument('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
    parser.add_argument('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          choices=['ddpmpp', 'ncsnpp', 'adm'], default='ddpmpp')
    parser.add_argument('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       choices=['vp', 've', 'edm'], default='edm')

    # Hyperparameters.
    parser.add_argument('--duration',      help='Training duration', metavar='MIMG',                          type=FloatRange(min=0, min_open=True), default=200)
    parser.add_argument('--batch',         help='Total batch size', metavar='INT',                            type=IntRange(min=1), default=128)
    parser.add_argument('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=IntRange(min=1))
    parser.add_argument('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
    parser.add_argument('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
    parser.add_argument('--lr',            help='Learning rate', metavar='FLOAT',                             type=FloatRange(min=0, min_open=True), default=10e-4)
    parser.add_argument('--ema',           help='EMA half-life', metavar='MIMG',                              type=FloatRange(min=0), default=0.5)
    parser.add_argument('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=FloatRange(min=0, max=1), default=0.13)
   
    # Performance-related.
    parser.add_argument('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False)
    parser.add_argument('--ls',            help='Loss scaling', metavar='FLOAT',                              type=FloatRange(min=0, min_open=True), default=1)
    parser.add_argument('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True)
    
    # I/O-related.
    parser.add_argument('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
    parser.add_argument('--nosubdir',      help='Do not create a subdirectory for results',                   action='store_true')
    parser.add_argument('--tick',          help='How often to print progress', metavar='KIMG',                type=IntRange(min=1), default=50)
    parser.add_argument('--snap',          help='How often to save snapshots', metavar='TICKS',               type=IntRange(min=1), default=50)
    parser.add_argument('--dump',          help='How often to dump state', metavar='TICKS',                   type=IntRange(min=1), default=500)
    parser.add_argument('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int, default=0)
    parser.add_argument('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
    parser.add_argument('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
    parser.add_argument('-n', '--dry-run', help='Print training options and exit',                            action='store_true')

    args = parser.parse_args()
    hyperparams = vars(args)
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, hyperparams), nprocs=world_size)

#----------------------------------------------------------------------------
