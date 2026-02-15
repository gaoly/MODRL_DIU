import torch

import os
from logging import getLogger

from MODFJSPEnv import KPEnv as Env
from MODFJSPModel import KPModel as Model

from MODFJSPProblemDef import get_random_problems

from einops import rearrange

from utils.utils import *


class KPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 PS,
                 EC_PROC,
                 EC_IDLE):

        
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.PS = PS
        self.EC_PROC = EC_PROC
        self.EC_IDLE = EC_IDLE
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(self.PS, self.EC_PROC, self.EC_IDLE, **self.env_params)
        self.model = Model(self.PS, self.EC_PROC, **self.model_params)
        
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint_mokp-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref, arrival_time, due_time):
        self.time_estimator.reset()
    
        score_AM = {}
      
        # 3 objs
        for i in range(3):
            score_AM[i] = AverageMeter()
          
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        
        while episode < test_num_episode:
            
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score = self._test_one_batch(shared_problem, pref, batch_size, episode, arrival_time, due_time)
            
            # 3 objs
            for i in range(3):
                score_AM[i].update(score[i], batch_size)
               
            episode += batch_size
    
            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f}, AUG_OBJ_3 SCORE: {:.4f} ".format(score_AM[0].avg, score_AM[1].avg, score_AM[2].avg))
                
                return [score_AM[0].avg.cpu(), score_AM[1].avg.cpu(), score_AM[2].avg.cpu()]
                
    def _test_one_batch(self, shared_probelm, pref, batch_size, episode, arrival_time, due_time):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        step = 0
        self.env.batch_size = batch_size
        self.env.problems = shared_probelm[episode: episode + batch_size][:, None,:,:,:].repeat_interleave(repeats=self.env_params['pomo_size'], dim=1)

        self.env.arrival_time = arrival_time
        self.env.due_time = due_time
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
        
        self.model.eval()
        # with torch.no_grad():
        reset_state, step_state, _, _ = self.env.reset()
        self.model.decoder.assign(pref, step_state, step)
        self.model.pre_forward(reset_state, step_state)
        self.model.pre_forward_period(step_state)
        self.model.set_gru()

        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, _ = self.model(state, reset_state)
            state, reward, done, arrival, Cmax, MU, DT = self.env.step(selected.clone())
            step += 1
            self.model.decoder.assign(pref, step_state, step)
            # 下面两行顺序固定
            if arrival:
                self.model.pre_forward(reset_state, step_state)
            self.model.set_gru_period(step_state)
            self.model.pre_forward_period(step_state)


        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward
        z = torch.ones(reward.shape).cuda() * 0.0
        tch_reward = pref * (reward - z)     
        tch_reward , _ = tch_reward.max(dim = 2)
        
        # set back reward and group_reward to negative
        reward = -reward
        tch_reward = -tch_reward
        
        group_reward = tch_reward
        _ , max_idx = group_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = Cmax.gather(1, max_idx)
        max_reward_obj2 = (MU).gather(1, max_idx)
        max_reward_obj3 = DT.gather(1, max_idx)
    
        score = []
        
        score.append(max_reward_obj1.float().mean())
        score.append(max_reward_obj2.float().mean())
        score.append(max_reward_obj3.float().mean())
    
        return score

     

