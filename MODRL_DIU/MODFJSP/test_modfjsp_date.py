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
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src

from MODFJSPTester import KPTester as Tester
from MODFJSPProblemDef import get_random_problems, get_random_problems_test
##########################################################################################
import time
from pymoo.indicators.hv import HV
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
problem_size = 30
pomo_size = 5
machine_size = 5
batch_size = 1
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

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train_modfjsp_full_m5_t100',  # directory path of pre-trained model and log files saved.
        'epoch': 100 # epoch version of pre-trained model to laod.
    },
    'test_episodes': 1,
    'test_batch_size': batch_size,
    'augmentation_enable': True,
    'aug_factor': 1, 
    'aug_batch_size': 100 
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_kp_n100',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 1


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

def get_pareto_solutions(result):
    obj = result.tolist()
    # print(obj)
    temp = []
    for inv in obj:
        if len(temp) == 0:
            temp.append(inv)
            continue
        flag = 1
        for i in range(len(temp)):
            if len(temp) <= i:
                break
            if temp[i][0] <= inv[0] and temp[i][1] <= inv[1] and temp[i][2] <= inv[2]:
                if temp[i][0] == inv[0] and temp[i][1] == inv[1] and temp[i][2] == inv[2]:
                    del temp[i]
                    i -= 1
                    continue
                flag = 0
                break
            if temp[i][0] >= inv[0] and temp[i][1] >= inv[1] and temp[i][2] >= inv[2]:
                del temp[i]
                i -= 1
        if flag == 1:
            temp.append(inv)
    return temp

def matrix_to_text(job_length, op_pt, op_per_mch, arrival_time , EL, due_time, filepath, directory):
    """
        Convert matrix form of the data into test form
    :param job_length: the number of operations in each job (shape [J])
    :param op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    :param op_per_mch: the average number of compatible machines of each operation
    :return: the standard text form of the instance
    """
    n_j = job_length.shape[0]
    n_op, n_m = op_pt.shape
    text = [f'{n_j}\t{n_m}\t{op_per_mch}']
    line = f''
    for j in range(n_j):
        line = line + ' ' + str(EL[j]) + ' ' + str(arrival_time[j]) + ' ' + str(due_time[j])
    text.append(line)

    op_idx = 0
    for j in range(n_j):
        line = f'{job_length[j]}'
        for _ in range(job_length[j]):
            use_mch = np.where(op_pt[op_idx] != 0)[0]
            line = line + ' ' + str(use_mch.shape[0])
            for k in use_mch:
                line = line + ' ' + str(k + 1) + ' ' + str(op_pt[op_idx][k])
            op_idx += 1
        text.append(line)

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 检查文件是否存在，存在则删除
    if os.path.exists(filepath):
        os.remove(filepath)

    doc = open(filepath, 'w')
    for i in range(len(text)):
        print(text[i], file=doc)
    doc.close()


##########################################################################################
def main(n_sols = 105):
    
    # timer_start = time.time()
    # logger_start = time.time()
    #
    # if DEBUG_MODE:
    #     _set_debug_mode()
    #
    # create_logger(**logger_params)
    # _print_config()
    #
    # tester = Tester(env_params=env_params,
    #                 model_params=model_params,
    #                 tester_params=tester_params)
    #
    # copy_all_src(tester.result_folder)
    #
    # sols = np.zeros([n_sols, 3])
    # if n_sols == 105:
    #     uniform_weights = torch.Tensor(das_dennis(13,3))  # 105
    # elif n_sols == 1035:
    #     uniform_weights = torch.Tensor(das_dennis(44,3))   # 1035
    # elif n_sols == 10011:
    #     uniform_weights = torch.Tensor(das_dennis(140,3))   # 10011

    #数据生成
    problem_size_list = [30,100,200]
    operation_size_list = [5,10,15]
    machine_size_list = [5,10,15]
    for o in operation_size_list:
        for j in problem_size_list:
            for m in machine_size_list:
                for instance_idx in range(20):
                    directory = './data2/fjs/fjs_n%d_m%d_o%d/'%(j,m,o)
                    filename = 'problem_n%d_m%d_o%d_i%d.fjs'%(j,m,o,instance_idx+1)
                    filepath = os.path.join(directory, filename)

                    shared_problem, arrival_time, due_time, op_pt_f, arrival_time_f, EL_f, due_time_f = get_random_problems_test(1, j, o, m)
                    op_per_mch_f = m
                    job_length_f = np.full(j, o)
                    matrix_to_text(job_length_f, op_pt_f.cpu().numpy(), op_per_mch_f, arrival_time_f.cpu().numpy(), EL_f.cpu().numpy(), due_time_f.cpu().numpy(), filepath, directory)

                    directory_pt = './data2/pt/pt_n%d_m%d_o%d/' % (j, m, o)
                    # 检查目录是否存在，不存在则创建
                    if not os.path.exists(directory_pt):
                        os.makedirs(directory_pt)
                    torch.save(shared_problem, './data2/pt/pt_n%d_m%d_o%d/shared_problem_n%d_m%d_o%d_i%d.pt'%(j,m,o,j,m,o,instance_idx+1))
                    torch.save(arrival_time, './data2/pt/pt_n%d_m%d_o%d/arrival_time_n%d_m%d_o%d_i%d.pt'%(j,m,o,j,m,o,instance_idx+1))
                    torch.save(due_time, './data2/pt/pt_n%d_m%d_o%d/due_time_n%d_m%d_o%d_i%d.pt'%(j,m,o,j,m,o,instance_idx+1))


##########################################################################################
if __name__ == "__main__":
    main()