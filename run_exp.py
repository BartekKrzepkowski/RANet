#!/usr/bin/env python3
import numpy as np
import torch

from src.utils.prepare import prepare_model, prepare_loaders, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective(lr, wd):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # model
    NUM_CLASSES = 10
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.block_step = 2
    args.stepmode = 'even'
    args.use_valid = False
    args.out_channels = 16


    args.nBlocks = 2
    args.Block_base = 2 # Block_base vs block
    args.step = 4
    args.stepmode ='even'
    args.compress_factor = 0.25
    args.nChannels = 64 # n = number? stała liczba kanałów we wszystkich warstwach?
    args.data = 'cifar10'
    args.growthRate = 16 # growth rate of what exactly?

    args.reduction = 0.5 # reduction of what?

    args.bottlenectFactor = '4-2-2-1' # bn = bottleneck?
    args.scale_list = '1-2-3-4'
    args.growFactor = '4-2-2-1' # gr = growth?


    args.bottlenectFactor = list(map(int, args.bottlenectFactor.split('-')))
    args.scale_list = list(map(int, args.scale_list.split('-')))
    args.growFactor = list(map(int, args.growFactor.split('-')))
    args.nScales = len(args.growFactor)

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    args.num_classes = NUM_CLASSES
    # trainer & scheduler
    RANDOM_SEED = 71
    EPOCHS = 200
    GRAD_ACCUM_STEPS = 1
    CLIP_VALUE = 0.0
    EXP_NAME = f'ran_lr_{lr}_wd_{wd}'
    PROJECT_NAME = 'RANets'

    # prepare params
    type_names = {
        'model_name': 'ran',
        'criterion_name': 'cls',
        'dataset_name': 'cifar10',
        'optim_name': 'sgd',
        'scheduler_name': 'cosine'
    }
    h_params_overall = {
        'model': {'args': args, 'criterion': None},
        'criterion': {'criterion_name': 'ce'},
        'dataset': {'dataset_path': 'data/', 'whether_aug': True},
        'loaders': {'batch_size': 256, 'pin_memory': True, 'num_workers': 6},
        'optim': {'lr': lr, 'momentum': 0.9, 'weight_decay': wd},
        'scheduler': {'eta_min': 1e-6, 'T_max': None},
        'type_names': type_names
    }
    # set seed to reproduce the results in the future
    manual_seed(random_seed=RANDOM_SEED, device=device)
    # prepare criterion
    criterion = prepare_criterion(type_names['criterion_name'], h_params_overall['criterion'])
    # prepare model
    h_params_overall['model']['criterion'] = criterion
    model = prepare_model(type_names['model_name'], model_params=h_params_overall['model']).to(device)
    # prepare loaders
    loaders = prepare_loaders(type_names['dataset_name'], h_params_overall['dataset'], h_params_overall['loaders'])
    # prepare optimizer & scheduler
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * EPOCHS
    h_params_overall['scheduler']['T_max'] = T_max
    optim, lr_scheduler = prepare_optim_and_scheduler(model, type_names['optim_name'], h_params_overall['optim'],
                                                      type_names['scheduler_name'], h_params_overall['scheduler'])

    # prepare trainer
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
    }
    trainer = TrainerClassification(**params_trainer)

    # prepare run
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    config = TrainerContext(
        epoch_start_at=0,
        epoch_end_at=EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        save_multi=T_max // 10,
        log_multi=(T_max // EPOCHS) // 10,
        clip_value=CLIP_VALUE,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'tensorboard', 'project_name': PROJECT_NAME, 'entity': 'ideas_cv',
                       'hyperparameters': h_params_overall, 'whether_use_wandb': True,
                       'layout': ee_tensorboard_layout(params_names), 'mode': 'online'
                       },
        whether_disable_tqdm=True,
        random_seed=RANDOM_SEED,
        device=device
    )
    trainer.run_exp(config)



if __name__ == "__main__":
        objective(5e-2, 1e-3)
