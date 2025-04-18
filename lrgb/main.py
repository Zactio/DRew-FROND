import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
try:
    from torch_geometric.graphgym.optimizer import create_optimizer, \
        create_scheduler, OptimizerConfig, SchedulerConfig
except:
    from torch_geometric.graphgym.optim import create_optimizer, \
        create_scheduler, OptimizerConfig, SchedulerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from graphgps.drew_utils import custom_set_out_dir
from param_calcs import set_d_fixed_params

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger

# to directly import your model (optional):
from graphgps.stage.drew_frond import DRewFrondModel
from graphgps.register import add_custom_model_cfg # [VERIFY] IF NEED - this is to register all of these keys in the model namespace to prevent Non-existent config key error

# from run_config import parser

# Optimizer/scheduler helper functions
try:
    from torch_geometric.graphgym.optimizer import create_optimizer, create_scheduler, OptimizerConfig, SchedulerConfig
except:
    from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig, SchedulerConfig


def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum
    )


def new_scheduler_config(cfg):
    return SchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch
    )


def custom_set_run_dir(cfg, run_id):
    """
    Custom output directory naming for each experiment run.
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings(args):
    """
    Determine how many times to run an experiment and with what settings.
    """
    if len(cfg.run_multiple_splits) == 0:
        # Multi-seed run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # Multi-split run mode
        if args.repeat != 1:
            raise NotImplementedError("Repeats with multiple splits not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices

    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Step 1: Parse command-line args (--cfg etc.)
    args = parse_args()   

    # Step 2: Initialize the base config node (cfg)
    set_cfg(cfg)

    # Step 3: Register any custom config fields (model.hidden_dim, etc.)
    add_custom_model_cfg()

    # Step 4: Register run_dir if you reference it in YAML (optional)
    cfg.run_dir = cfg.out_dir

    # Step 5: Load YAML config into cfg (includes overwriting with command-line args)
    load_cfg(cfg, args)

    # Step 6: Set any fixed parameters (optional/custom logic)
    set_d_fixed_params(cfg)

    # Step 7: Set the output directory (e.g., results/run_xxx/)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)

    # Step 8: Save the final cfg as a YAML to the run directory
    dump_cfg(cfg)

    # Step 9: Set PyTorch threads (useful for CPU scaling)
    torch.set_num_threads(cfg.num_threads)

    # Step 10: Run multiple iterations (seeds or splits)
    for run_id, seed, split_index in zip(*run_loop_settings(args)):
        # Configure each run's directory
        custom_set_run_dir(cfg, run_id)

        # Enable logging to both console and file
        set_printing()

        # Set split/seed/run_id
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id

        # Set random seeds
        seed_everything(cfg.seed)

        # Auto-select device (CPU/GPU)
        auto_select_device()

        # Load fine-tuning checkpoint config if applicable
        if cfg.train.finetune:
            cfg = load_pretrained_model_cfg(cfg)

        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")

        # === Data loaders and logger setup ===
        loaders = create_loader()
        loggers = create_logger()

        # === Instantiate the model ===
        dataset = loaders[0].dataset
        
        model = DRewFrondModel(cfg, dataset).to(cfg.device)

        # Fine-tune from checkpoint if applicable
        if cfg.train.finetune:
            model = init_model_from_pretrained(model, cfg.train.finetune, cfg.train.freeze_pretrained)

        # === Optimizer and Scheduler ===
        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        # === Log model and config ===
        logging.info(model)
        logging.info(cfg)

        cfg.params = params_count(model)
        logging.info(f"Num parameters: {cfg.params}")

        # === Training loop ===
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with 'standard' train.mode; consider 'custom'")
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            # Custom train mode
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Step 11: Aggregate results across seeds/splits
    agg_runs(cfg.out_dir, cfg.metric_best)

    # Step 12: Mark YAML as done (optional batch-run behavior)
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')

    logging.info(f"[*] All done: {datetime.datetime.now()}")



# def new_optimizer_config(cfg):
#     return OptimizerConfig(optimizer=cfg.optim.optimizer,
#                            base_lr=cfg.optim.base_lr,
#                            weight_decay=cfg.optim.weight_decay,
#                            momentum=cfg.optim.momentum)


# def new_scheduler_config(cfg):
#     return SchedulerConfig(scheduler=cfg.optim.scheduler,
#                            steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
#                            max_epoch=cfg.optim.max_epoch)


# def custom_set_run_dir(cfg, run_id):
#     """Custom output directory naming for each experiment run.

#     Args:
#         cfg (CfgNode): Configuration node
#         run_id (int): Main for-loop iter id (the random seed or dataset split)
#     """
#     cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
#     # Make output directory
#     if cfg.train.auto_resume:
#         os.makedirs(cfg.run_dir, exist_ok=True)
#     else:
#         makedirs_rm_exist(cfg.run_dir)


# def run_loop_settings():
#     """Create main loop execution settings based on the current cfg.

#     Configures the main execution loop to run in one of two modes:
#     1. 'multi-seed' - Reproduces default behaviour of GraphGym when
#         args.repeats controls how many times the experiment run is repeated.
#         Each iteration is executed with a random seed set to an increment from
#         the previous one, starting at initial cfg.seed.
#     2. 'multi-split' - Executes the experiment run over multiple dataset splits,
#         these can be multiple CV splits or multiple standard splits. The random
#         seed is reset to the initial cfg.seed value for each run iteration.

#     Returns:
#         List of run IDs for each loop iteration
#         List of rng seeds to loop over
#         List of dataset split indices to loop over
#     """
#     if len(cfg.run_multiple_splits) == 0:
#         # 'multi-seed' run mode
#         num_iterations = args.repeat
#         seeds = [cfg.seed + x for x in range(num_iterations)]
#         split_indices = [cfg.dataset.split_index] * num_iterations
#         run_ids = seeds
#     else:
#         # 'multi-split' run mode
#         if args.repeat != 1:
#             raise NotImplementedError("Running multiple repeats of multiple "
#                                       "splits in one run is not supported.")
#         num_iterations = len(cfg.run_multiple_splits)
#         seeds = [cfg.seed] * num_iterations
#         split_indices = cfg.run_multiple_splits
#         run_ids = split_indices
#     return run_ids, seeds, split_indices


# if __name__ == '__main__':
#     # Load cmd line args
#     args = parser.parse_args()
#     opt = vars(args)
#     # Load config file
#     set_cfg(cfg)
#     cfg.run_dir = cfg.out_dir # prevents error when loading config.yaml file generated from run
#     load_cfg(cfg, args)
#     set_d_fixed_params(cfg)
#     custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
#     dump_cfg(cfg)
#     # Set Pytorch environment
#     torch.set_num_threads(cfg.num_threads)
#     # Repeat for multiple experiment runs
#     for run_id, seed, split_index in zip(*run_loop_settings()):
#         # Set configurations for each run
#         custom_set_run_dir(cfg, run_id)
#         set_printing()
#         cfg.dataset.split_index = split_index
#         cfg.seed = seed
#         cfg.run_id = run_id
#         seed_everything(cfg.seed)
#         auto_select_device()
#         if cfg.train.finetune:
#             cfg = load_pretrained_model_cfg(cfg)
#         logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
#                      f"split_index={cfg.dataset.split_index}")
#         logging.info(f"    Starting now: {datetime.datetime.now()}")
#         # Set machine learning pipeline
#         loaders = create_loader()
#         loggers = create_logger()
#         #model = create_model()
#         model = DRewFrondModel(opt, dataset, device).to(device)
#         if cfg.train.finetune:
#             model = init_model_from_pretrained(model, cfg.train.finetune,
#                                                cfg.train.freeze_pretrained)
#         optimizer = create_optimizer(model.parameters(),
#                                      new_optimizer_config(cfg))
#         scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
#         # Print model info
#         logging.info(model)
#         logging.info(cfg)
#         cfg.params = params_count(model)
#         logging.info('Num parameters: {}'.format(cfg.params))
#         # Start training
#         if cfg.train.mode == 'standard':
#             if cfg.wandb.use:
#                 logging.warning("[W] WandB logging is not supported with the "
#                                 "default train.mode, set it to `custom`")
#             train(loggers, loaders, model, optimizer, scheduler)
#         else:
#             train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
#                                        scheduler)
#     # Aggregate results from different seeds

#     agg_runs(cfg.out_dir, cfg.metric_best)

#     # When being launched in batch mode, mark a yaml as done
#     if args.mark_done:
#         os.rename(args.cfg_file, '{}_done'.format(args.cfg_file))
#     logging.info(f"[*] All done: {datetime.datetime.now()}")
