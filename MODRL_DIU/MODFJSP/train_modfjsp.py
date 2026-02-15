##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
import torch
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src

from MODFJSPTrainer import TSPTrainer as Trainer

##########################################################################################
# parameters
problem_size = 30
pomo_size = 5
machine_size = 5
batch_size = 8
operation_size = 5
env_params = {
    'problem_size': problem_size,
    'pomo_size': pomo_size,
    'operation_size': operation_size,
    'machine_size': machine_size,

}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'feature_size': machine_size+3,
    'machine_size': machine_size,
    'problem_size': problem_size,
    'operation_size': operation_size,
    'pomo_size': pomo_size,
    'train_batch_size': batch_size,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4, 
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [20,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 2000,
    'train_episodes': 25008,
    'train_batch_size': batch_size,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_fjsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/20250426_185142_train_fjsp_n100_t5',  # directory path of pre-trained model and log files saved.
        'epoch': 5,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_fjsp_n100',
        'filename': 'run_log'
    }
}

##########################################################################################
# main
def main():
    if machine_size == 5:
        PS = torch.tensor([1, 1, 1.2, 1.5, 1.5], device='cuda:0', requires_grad=False)
        EC_PROC = torch.tensor([0.4141, 0.3905, 0.5252, 0.8215, 0.6464], device='cuda:0', requires_grad=False)
        EC_IDLE = torch.tensor([0.0774, 0.0606, 0.0909, 0.1313, 0.1077], device='cuda:0', requires_grad=False)
    elif machine_size == 10:
        PS = torch.tensor([1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.5], device='cuda:0', requires_grad=False)
        EC_PROC = torch.tensor([0.4141, 0.3905, 0.3872, 0.5252, 0.5084, 0.4074, 0.5959, 0.8215, 0.6464, 0.7710], device='cuda:0', requires_grad=False)
        EC_IDLE = torch.tensor([0.0774, 0.0606, 0.0639, 0.0909, 0.1010, 0.0976, 0.1077, 0.1313, 0.1077, 0.1313], device='cuda:0', requires_grad=False)
    elif machine_size == 15:
        PS = torch.tensor([1, 1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.5, 1.5, 1.5], device='cuda:0', requires_grad=False)
        EC_PROC = torch.tensor([0.4141, 0.3905, 0.3872, 0.4646, 0.3501, 0.5252, 0.5084, 0.4074, 0.5959, 0.6599, 0.8215, 0.6464, 0.7710, 0.6599, 1.0], device='cuda:0', requires_grad=False)
        EC_IDLE = torch.tensor([0.0774, 0.0606, 0.0639, 0.0774, 0.0740, 0.0909, 0.1010, 0.0976, 0.1077, 0.1043, 0.1313, 0.1077, 0.1313, 0.1346, 0.1346], device='cuda:0', requires_grad=False)
    PS = PS.unsqueeze(0).unsqueeze(0).repeat(batch_size, problem_size, operation_size, 1)
    EC_PROC = EC_PROC.unsqueeze(0).unsqueeze(0).repeat(batch_size, pomo_size, 1)
    EC_IDLE = EC_IDLE.unsqueeze(0).unsqueeze(0).repeat(batch_size, pomo_size, 1)

    
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      PS=PS,
                      EC_PROC=EC_PROC,
                      EC_IDLE = EC_IDLE)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()