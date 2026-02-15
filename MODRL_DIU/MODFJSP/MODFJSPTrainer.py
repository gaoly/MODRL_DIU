import torch
from logging import getLogger

from MODFJSPEnv import KPEnv as Env
from MODFJSPModel import KPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

import numpy as np

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 PS,
                 EC_PROC,
                 EC_IDLE):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.PS = PS
        self.EC_PROC = EC_PROC
        self.EC_IDLE = EC_IDLE
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(self.PS, self.EC_PROC, **self.model_params)
        
        print(sum(p.numel() for p in self.model.parameters()))
        print(sum(p.numel() for p in self.model.encoder.parameters()))
        
        self.env = Env(self.PS, self.EC_PROC, self.EC_IDLE, **self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_mokp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score_obj1, train_score_obj2, train_score_obj3, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_score_obj3', epoch, train_score_obj3)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_mokp-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
        score_AM_obj3 = AverageMeter()
    
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score_obj1, avg_score_obj2, avg_score_obj3, avg_loss = self._train_one_batch(batch_size, episode)
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            score_AM_obj3.update(avg_score_obj3, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f}, Obj3 Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM_obj1.avg, score_AM_obj2.avg, score_AM_obj3.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f}, Obj3 Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, score_AM_obj3.avg, loss_AM.avg))

        return score_AM_obj1.avg, score_AM_obj2.avg, score_AM_obj3.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, episode):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        step = 0
        # pref = torch.rand([2])
        # # pref = torch.ones([2])
        # # pref[1] = 0
        # pref = pref / torch.sum(pref)
        pref = torch.rand([3])
        pref = pref / torch.sum(pref)

        reset_state, step_state, _, _ = self.env.reset()
        
        self.model.decoder.assign(pref, step_state, step)
        self.model.pre_forward(reset_state, step_state)
        self.model.pre_forward_period(step_state)
        self.model.set_gru()
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
       
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state, reset_state)
            state, reward, done, arrival, Cmax, MU, DT = self.env.step(selected.clone())
            step += 1
            self.model.decoder.assign(pref, step_state, step)
            #下面两行顺序固定
            if arrival:
                self.model.pre_forward(reset_state, step_state)
            self.model.set_gru_period(step_state)
            self.model.pre_forward_period(step_state)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        ###############################################
        # KP is to maximize the reward, here we set it to inverse to be minimized
        #reward = idle_time + makespan
        reward = - reward
        tch_reward = (pref * reward).sum(dim=2)
        # set back reward and group_reward to positive and to maximize
        tch_reward = -tch_reward
        
        log_prob = prob_list.log().sum(dim=2)


        tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)


        tch_loss = tch_advantage * log_prob # Minus Sign
       
        loss_mean = tch_loss.mean()
       
     
        # Score
        ###############################################
        _ , max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = Cmax.gather(1, max_idx)
        max_reward_obj2 = MU.gather(1, max_idx)
        max_reward_obj3 = DT.gather(1, max_idx)

        score_mean_obj1 = max_reward_obj1.float().mean()
        score_mean_obj2 = max_reward_obj2.float().mean()
        score_mean_obj3 = max_reward_obj3.float().mean()
        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        return score_mean_obj1.item(), score_mean_obj2.item(), score_mean_obj3.item(), loss_mean.item()