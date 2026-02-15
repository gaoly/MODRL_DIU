from dataclasses import dataclass
import torch
import numpy as np
import random
from MODFJSPProblemDef import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    pomo_size: int
    machine_size: int
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    selected_count: int = 0
    current_jobnum: int = 2
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None

    job_available_time: torch.Tensor = None
    job_arrival_time: torch.Tensor = None
    job_course: torch.Tensor = None
    machine_available_time: torch.Tensor = None
    machine_busy_time: torch.Tensor = None
    selected_job: torch.Tensor = None
    selected_job_fake: torch.Tensor = None
    selected_machine: torch.Tensor = None
    op_idx: torch.Tensor = None
    op_idx_RT: torch.Tensor = None
    course_idx: torch.Tensor = None
    complete_job_mask: torch.Tensor = None
    dynamic_global_info: torch.Tensor = None
    # shape: (batch, pomo, node)


class KPEnv:
    def __init__(self, PS, EC_PROC, EC_IDLE, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.operation_size = env_params['operation_size']
        self.machine_size = env_params['machine_size']
        self.PS = PS
        self.EC_PROC = EC_PROC
        self.EC_IDLE = EC_IDLE
        # Const @Load_Problem
        ####################################
        self.current_jobnum = None
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        self.problems, self.arrival_time, self.due_time = get_random_problems(batch_size, self.problem_size,
                                                                              self.operation_size, self.machine_size, self.PS, self.EC_PROC)
        self.current_jobnum = 5
        # problems.shape: (batch, job_size, operation_size, 1)
        if aug_factor > 1:
            raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        # MOKP
        ###################################
        # self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size+1, 3)))
        # self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.problems = self.problems[:, None, :, :, :].repeat_interleave(repeats=self.pomo_size, dim=1)
        # if self.problem_size == 50:
        #     capacity = 12.5
        # elif self.problem_size == 100:
        #     capacity = 25
        # elif self.problem_size == 200:
        #     capacity = 25
        # else:
        #     raise NotImplementedError
        self.machine_available_time = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.machine_size)))
        self.machine_busy_time = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.machine_size)))
        self.job_course = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.current_jobnum))).int()
        # self.job_course[:, :, 0] = self.operation_size
        self.energy_consumption = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))

        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))

        self.ninf_mask = torch.zeros(self.batch_size, self.pomo_size, self.current_jobnum * self.machine_size)
        # self.ninf_mask[:,:,0:self.machine_size] = -np.inf
        self.fit_ninf_mask = None
        self.finished = torch.BoolTensor(np.zeros((self.batch_size, self.pomo_size)))

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.current_jobnum = 5
        self.selected_job_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.selected_machine_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # MOKP
        ###################################
        temp1 = torch.Tensor(np.arange(0, self.current_jobnum)).int()
        temp1 = temp1[None, None, :, None, None].expand(self.batch_size, self.pomo_size, self.current_jobnum,
                                                        self.operation_size + 1, 1)
        temp2 = torch.Tensor(np.arange(0, self.operation_size + 1)).int()
        temp2 = temp2[None, None, None, :, None].expand(self.batch_size, self.pomo_size, self.current_jobnum,
                                                        self.operation_size + 1, 1)
        self.op_map = torch.cat((temp1, temp2), dim=-1)
        self.op_idx = torch.Tensor(
            np.zeros((self.batch_size, self.pomo_size, self.current_jobnum, (self.operation_size + 1) * 2))).int()
        self.course_idx = torch.Tensor(
            np.zeros((self.batch_size, self.pomo_size, self.current_jobnum, self.operation_size + 1))).int()
        self.op_idx[:, :, :, 0:self.operation_size + 1] = 1
        self.course_idx[:, :, :, 0] = 1
        self.op_idx_RT = self.op_idx.clone()

        self.complete_job_mask = torch.Tensor(np.ones((self.batch_size, self.pomo_size, self.current_jobnum))).int()

        self.machine_available_time = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.machine_size)))
        self.machine_busy_time = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.machine_size)))
        self.energy_consumption = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        # self.machine_available_time = torch.Tensor(np.random.rand(self.batch_size, self.pomo_size, self.machine_size))
        # self.machine_available_time = torch.Tensor(np.random.rand(self.batch_size, self.pomo_size, self.machine_size))
        self.job_course = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.current_jobnum))).int()
        # self.job_course[:,:,0] = self.operation_size
        self.job_available_time = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.current_jobnum)))
        self.job_arrival_time = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, self.current_jobnum)))
        # self.job_available_time[:, :, 0] = 999
        # if self.problem_size == 50:
        #     capacity = 12.5
        # elif self.problem_size == 100:
        #     capacity = 25
        # elif self.problem_size == 200:
        #     capacity = 25
        # else:
        #     raise NotImplementedError

        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))

        self.ninf_mask = torch.zeros(self.batch_size, self.pomo_size, self.current_jobnum * self.machine_size)
        # self.ninf_mask[:,:,0:self.machine_size] = -np.inf
        self.fit_ninf_mask = None
        self.finished = torch.BoolTensor(np.zeros((self.batch_size, self.pomo_size)))

        std_machine_available_time = torch.std(self.machine_available_time, -1, keepdim=True)
        std_machine_available_time = (std_machine_available_time+0.00001) / (torch.max(std_machine_available_time)+0.00001)
        std_job_course = torch.std(self.job_course.float(), -1, keepdim=True)
        std_job_course = (std_job_course+0.00001) / (torch.max(std_job_course)+0.00001)


        self.dynamic_global_info = torch.cat((std_machine_available_time,
                                              std_job_course,
                                              torch.mean(self.problems[:, :, :self.current_jobnum, 0, -2], dim=-1)[:, :,
                                              None]), dim=-1)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.dynamic_global_info = self.dynamic_global_info
        self.step_state.course_idx = self.course_idx
        self.step_state.job_arrival_time = self.job_arrival_time
        self.step_state.op_idx = self.op_idx
        self.step_state.op_idx_RT = self.op_idx_RT
        self.step_state.current_jobnum = self.current_jobnum
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.job_available_time = self.job_available_time
        self.step_state.machine_busy_time = self.machine_busy_time
        self.step_state.job_course = self.job_course
        self.step_state.machine_available_time = self.machine_available_time
        self.step_state.selected_count = self.selected_count
        self.step_state.complete_job_mask = self.complete_job_mask
        reward = None
        done = False
        return Reset_State(self.problems, self.pomo_size, self.machine_size), self.step_state, reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo) (job * machine)
        self.selected_count+=1
        op_map = self.op_map.reshape(self.batch_size, self.pomo_size, -1, 2)
        op_map = op_map[self.op_idx[:, :, :, 0:(int(self.op_idx.size(-1) / 2))].reshape(self.batch_size, self.pomo_size,
                                                                                        -1) == 1].reshape(
            self.batch_size, self.pomo_size, -1, 2)
        course_idx = self.course_idx[:, :, :, 0:self.operation_size + 1].reshape(self.batch_size, self.pomo_size, -1)
        course_idx = course_idx[
            self.op_idx[:, :, :, 0:(int(self.op_idx.size(-1) / 2))].reshape(self.batch_size, self.pomo_size,
                                                                            -1) == 1].reshape(self.batch_size,
                                                                                              self.pomo_size, -1)
        op_map = op_map[course_idx == 1].reshape(self.batch_size, self.pomo_size, -1, 2)

        # 工件索引需要通过映射表计算获得
        self.selected_job_fake = torch.floor(selected.int() / (self.machine_size)).int()
        # 有问题，需要删除op_map中做完的记录
        self.selected_job = op_map[self.BATCH_IDX, self.POMO_IDX, self.selected_job_fake][:, :, 0]
        self.selected_machine = selected.int() % (self.machine_size)
        self.selected_job_list = torch.cat((self.selected_job_list, self.selected_job[:, :, None]), dim=2)
        self.selected_machine_list = torch.cat((self.selected_machine_list, self.selected_machine[:, :, None]), dim=2)

        # 更新course_idx和op_idx
        self.course_idx[self.BATCH_IDX, self.POMO_IDX, self.selected_job] = self.course_idx[
            self.BATCH_IDX, self.POMO_IDX, self.selected_job].roll(1, dims=-1)
        # self.course_idx[:,:,:,-1] = 0
        self.op_idx_RT[self.BATCH_IDX, self.POMO_IDX, self.selected_job] = self.op_idx_RT[
            self.BATCH_IDX, self.POMO_IDX, self.selected_job].roll(1, dims=-1)
        # problems (batch_size, problem_size, operation_size, machine_size * 2)
        problems = self.problems
        temp_data = problems[self.BATCH_IDX, self.POMO_IDX, self.selected_job]
        temp_data = temp_data[
            self.BATCH_IDX, self.POMO_IDX, self.job_course[self.BATCH_IDX, self.POMO_IDX, self.selected_job]]
        proc = temp_data[self.BATCH_IDX, self.POMO_IDX, self.selected_machine]
        # enc = temp_data[self.BATCH_IDX, self.POMO_IDX, self.selected_machine + self.machine_size]

        # self.energy_consumption += enc
        self.machine_busy_time[self.BATCH_IDX, self.POMO_IDX, self.selected_machine] += proc
        self.job_available_time[self.BATCH_IDX, self.POMO_IDX, self.selected_job] = torch.where(
            self.job_available_time[self.BATCH_IDX, self.POMO_IDX, self.selected_job] > self.machine_available_time[
                self.BATCH_IDX, self.POMO_IDX, self.selected_machine],
            self.job_available_time[self.BATCH_IDX, self.POMO_IDX, self.selected_job],
            self.machine_available_time[self.BATCH_IDX, self.POMO_IDX, self.selected_machine]) + proc

        self.machine_available_time[self.BATCH_IDX, self.POMO_IDX, self.selected_machine] = self.job_available_time[
            self.BATCH_IDX, self.POMO_IDX, self.selected_job]
        self.job_course[self.BATCH_IDX, self.POMO_IDX, self.selected_job] += 1

        

        # 恢复虚拟节点
        # self.job_course[:, :, 0] = self.operation_size
        # self.job_available_time[:, :, 0] = 999

        # 确定工件是否到达,隐藏了两个判断逻辑，只要有一个算例在工件到达前无事可做，或者所有当前工件都调度完时，工件到达
        ninf_mask1 = torch.where(self.job_course == self.operation_size, 0, 1)
        temp1 = torch.where(self.job_available_time < self.arrival_time[0, self.current_jobnum - 5], 1, 0)
        temp1 = temp1 * ninf_mask1
        temp2 = torch.where(self.machine_available_time < self.arrival_time[0, self.current_jobnum - 5], -np.inf, 0)
        temp1 = (temp1 == 1).any(dim=2)
        temp2 = (temp2 == -np.inf).any(dim=2)
        temp = torch.cat((temp1[:, :, None], temp2[:, :, None]), dim=-1)
        temp = (temp == True).all(dim=2)
        arrival = (temp == False).any()

        # 生成mask
        ninf_mask1 = torch.where(self.job_course == self.operation_size, -np.inf, 0)
        ninf_mask2 = torch.where(self.job_available_time > self.arrival_time[0, self.current_jobnum - 5], -np.inf, 0)
        self.ninf_mask = ninf_mask1 + ninf_mask2
        self.ninf_mask = self.ninf_mask[self.complete_job_mask == 1].reshape(self.batch_size, self.pomo_size,
                                                                             -1)
        # self.ninf_mask = ninf_mask2[ninf_mask1==0].reshape(self.batch_size, self.pomo_size, -1)
        ninf_mask3 = torch.where(self.machine_available_time > self.arrival_time[0, self.current_jobnum - 5], -np.inf,
                                 0)
        ninf_mask3 = ninf_mask3[:, :, None, :].repeat_interleave(repeats=self.ninf_mask.size(2), dim=2)
        self.ninf_mask = self.ninf_mask[:, :, :, None].repeat_interleave(repeats=self.machine_size, dim=-1)
        self.ninf_mask = self.ninf_mask + ninf_mask3
        self.ninf_mask = self.ninf_mask.reshape(self.batch_size, self.pomo_size, -1)

        self.finished = (ninf_mask1 == -np.inf).all(dim=2)
        done = self.finished.all() and (self.current_jobnum == self.problem_size)

        std_machine_available_time = torch.std(self.machine_available_time, -1, keepdim=True)
        std_machine_available_time = (std_machine_available_time+0.00001) / (torch.max(std_machine_available_time)+0.00001)
        std_job_course = torch.std(self.job_course.float(), -1, keepdim=True)
        std_job_course = (std_job_course+0.00001) / (torch.max(std_job_course)+0.00001)
        self.dynamic_global_info = torch.cat((std_machine_available_time,
                                              std_job_course,
                                              torch.mean(self.problems[:, :, :self.current_jobnum, 0, -2], dim=-1)[:, :,
                                              None]), dim=-1)
        self.step_state.dynamic_global_info = self.dynamic_global_info
        self.step_state.course_idx = self.course_idx
        self.step_state.op_idx = self.op_idx
        self.step_state.op_idx_RT = self.op_idx_RT
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.job_available_time = self.job_available_time
        self.step_state.machine_available_time = self.machine_available_time
        self.step_state.machine_busy_time = self.machine_busy_time
        self.step_state.job_course = self.job_course
        self.step_state.finished = self.finished
        self.step_state.selected_count = self.selected_count
        self.step_state.selected_job = self.selected_job
        self.step_state.selected_job_fake = self.selected_job_fake
        self.step_state.selected_machine = self.selected_machine

        reward = None
        Cmax = None
        ML = None
        DT = None
        if done:
            # print(idle_time[0, 0]) 2.9
            # print(reward[0, 0]) 3.2
            # print(DT[0, 0]) 9
            # Cmax
            Cmax, _ = torch.max(self.machine_available_time, dim=-1)
            output1 = Cmax
            # machine load
            ML = (self.machine_busy_time * self.EC_PROC).sum(dim=-1)
            #print(MU)
            idle_time = ((self.machine_available_time - self.machine_busy_time)*self.EC_IDLE).sum(dim=-1)
            ML = ML + idle_time
            output2 = ML/self.machine_size
            # DT
            temp = (self.job_available_time - self.job_arrival_time) - (self.problems[:, :, :, 0, -2] * 1.7 * self.problems[:,:,:,0:self.operation_size,0:self.machine_size].mean(dim=-1).sum(dim=-1))
            DT = torch.clamp(temp, min=0).sum(dim=-1)
            #DT = torch.where(temp>0,temp,0).sum(dim=-1)
            output3 = DT/self.machine_size
            reward = torch.cat((output1[:, :, None], output2[:, :, None], output3[:, :, None]), dim=2)


        else:
            reward = None

        arrival_flag = False
        if (self.current_jobnum < self.problem_size) and arrival:
            arrival_flag = True
            self.job_arrival()

        return self.step_state, reward, done, arrival_flag, Cmax, ML, DT

    def normalize(self, reward):
        max_val = torch.max(torch.max(reward,-1)[0],-1)[0].item()
        min_val = torch.min(torch.min(reward, -1)[0], -1)[0].item()
        # if min_val1 < min_val:
        #     min_val = min_val1
        reward_n = (reward - min_val)/(max_val - min_val)
        #reward_n = reward
        return reward_n

    def job_arrival(self):
        new_job_course = torch.Tensor(np.zeros((self.batch_size, self.pomo_size, 1))).int()
        # new_job_available_time = torch.Tensor(np.ones((self.batch_size, self.pomo_size, 1))) * self.arrival_time[0, self.current_jobnum-2]
        temp1, _ = torch.min(self.job_available_time, dim=-1)
        temp2, _ = torch.min(self.machine_available_time, dim=-1)

        temp = torch.cat((temp1[:, :, None], temp2[:, :, None]), dim=-1)
        new_job_available_time, _ = torch.max(temp, dim=-1)

        self.job_course = torch.cat((self.job_course, new_job_course), dim=-1)
        self.job_available_time = torch.cat((self.job_available_time, new_job_available_time[:, :, None]), dim=-1)
        self.job_arrival_time = torch.cat((self.job_arrival_time, new_job_available_time[:, :, None]), dim=-1)
        self.step_state.job_course = self.job_course
        self.step_state.job_available_time = self.job_available_time
        self.step_state.job_arrival_time = self.job_arrival_time
        self.current_jobnum += 1

        #更新arr_time
        max_arr,_ = torch.max(torch.max(torch.max(self.job_arrival_time,-1)[0],-1)[0],-1)
        temp = self.job_arrival_time/max_arr
        temp = temp[:,:,:,None].expand(self.batch_size,self.pomo_size,self.current_jobnum,self.operation_size)
        self.problems[:, :, 0:self.current_jobnum, 0:self.operation_size, -1] = temp

        # 更新op_map, op_idx, course_idx
        temp1 = torch.Tensor(np.arange(0, self.current_jobnum)).int()
        temp1 = temp1[None, None, :, None, None].expand(self.batch_size, self.pomo_size, self.current_jobnum,
                                                        self.operation_size + 1, 1)
        temp2 = torch.Tensor(np.arange(0, self.operation_size + 1)).int()
        temp2 = temp2[None, None, None, :, None].expand(self.batch_size, self.pomo_size, self.current_jobnum,
                                                        self.operation_size + 1, 1)
        self.op_map = torch.cat((temp1, temp2), dim=-1)
        new_op_idx = torch.Tensor(
            np.zeros((self.batch_size, self.pomo_size, 1, (self.operation_size + 1) * 2))).int()
        new_course_idx = torch.Tensor(
            np.zeros((self.batch_size, self.pomo_size, 1, self.operation_size + 1))).int()
        new_op_idx[:, :, :, 0:self.operation_size + 1] = 1
        new_course_idx[:, :, :, 0] = 1
        self.op_idx_RT = torch.cat((self.op_idx_RT, new_op_idx), dim=2)
        self.course_idx = torch.cat((self.course_idx, new_course_idx), dim=2)

        # 通过检测方法，找到每个批每个pomo中完成的工件数量，如果都有至少一个则想办法同时除籍，保证后续运转。mask也需要一个楔子做调整。
        min_val = (self.op_idx_RT[:, :, :, 0:self.operation_size] == 0).all(dim=-1).sum(dim=-1).min().item()
        self.complete_job_mask = torch.Tensor(np.ones((self.batch_size, self.pomo_size, self.current_jobnum))).int()
        if min_val > 0:
            for i in range(self.op_idx_RT.size(0)):
                for j in range(self.op_idx_RT.size(1)):
                    count = 0
                    for k in range(self.op_idx_RT.size(2)):
                        if (self.op_idx_RT[i, j, k, 0:self.operation_size] == 0).all(dim=-1).item() is True:
                            self.op_idx_RT[i, j, k, self.operation_size] = 1
                            self.course_idx[i, j, k, self.operation_size] = 1
                    for k in range(self.op_idx_RT.size(2)):
                        if (self.op_idx_RT[i, j, k, 0:self.operation_size] == 0).all(dim=-1).item() is True:
                            self.op_idx_RT[i, j, k] = 0
                            self.course_idx[i, j, k] = 0
                            self.complete_job_mask[i, j, k] = 0
                            count += 1
                        if count == min_val:
                            break
        self.op_idx = self.op_idx_RT.clone()
        # 只有工件到达才更新外部使用的op_idx
        self.step_state.op_idx = self.op_idx
        self.step_state.course_idx = self.course_idx
        self.step_state.op_idx_RT = self.op_idx_RT
        self.step_state.complete_job_mask = self.complete_job_mask

        # if self.current_jobnum == 11:
        #     print(new_job_available_time[0,0])
        self.step_state.current_jobnum = self.current_jobnum

        ninf_mask1 = torch.where(self.job_course == self.operation_size, -np.inf, 0)
        ninf_mask2 = torch.where(self.job_available_time > self.arrival_time[0, self.current_jobnum - 5], -np.inf, 0)
        ninf_mask3 = torch.where(self.machine_available_time > self.arrival_time[0, self.current_jobnum - 5], -np.inf,
                                 0)
        self.ninf_mask = ninf_mask1 + ninf_mask2
        self.ninf_mask = self.ninf_mask[self.complete_job_mask == 1].reshape(self.batch_size, self.pomo_size, -1)
        ninf_mask3 = ninf_mask3[:, :, None, :].repeat_interleave(repeats=self.ninf_mask.size(2), dim=2)

        self.ninf_mask = self.ninf_mask[:, :, :, None].repeat_interleave(repeats=self.machine_size, dim=-1)
        self.ninf_mask = self.ninf_mask + ninf_mask3
        self.ninf_mask = self.ninf_mask.reshape(self.batch_size, self.pomo_size, -1)
        self.step_state.ninf_mask = self.ninf_mask

        # 确定工件是否到达,隐藏了两个判断逻辑，只要有一个算例在工件到达前无事可做，或者所有当前工件都调度完时，工件到达
        ninf_mask1 = torch.where(self.job_course == self.operation_size, 0, 1)
        temp1 = torch.where(self.job_available_time < self.arrival_time[0, self.current_jobnum - 5], 1, 0)
        temp1 = temp1 * ninf_mask1
        temp2 = torch.where(self.machine_available_time < self.arrival_time[0, self.current_jobnum - 5], -np.inf, 0)
        temp1 = (temp1 == 1).any(dim=2)
        temp2 = (temp2 == -np.inf).any(dim=2)
        temp = torch.cat((temp1[:, :, None], temp2[:, :, None]), dim=-1)
        temp = (temp == True).all(dim=2)
        arrival = (temp == False).any()

        std_machine_available_time = torch.std(self.machine_available_time, -1, keepdim=True)
        std_machine_available_time = (std_machine_available_time+0.00001) / (torch.max(std_machine_available_time)+0.00001)
        std_job_course = torch.std(self.job_course.float(), -1, keepdim=True)
        std_job_course = (std_job_course+0.00001) / (torch.max(std_job_course)+0.00001)
        self.dynamic_global_info = torch.cat((std_machine_available_time,
                                              std_job_course,
                                              torch.mean(self.problems[:,:,:self.current_jobnum,0,-2], dim=-1)[:,:,None]), dim=-1)
        self.step_state.dynamic_global_info = self.dynamic_global_info
        if (self.current_jobnum < self.problem_size) and arrival:
            self.job_arrival()
            # print(self.current_jobnum)
            # print("工件连续到达")

