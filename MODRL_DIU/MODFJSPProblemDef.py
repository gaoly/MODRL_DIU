import torch
import numpy as np


def get_random_problems(batch_size, problem_size, operation_size, machine_size, PS, EC_PROC):
    proc = torch.randint(1, 100, size=(batch_size, problem_size, operation_size)).unsqueeze(3).repeat(1,1,1,machine_size)
    proc = proc/PS/100
    ec_proc = proc*EC_PROC[0,0][None,None,None,:]

    avg_proc = torch.mean(proc) * problem_size * operation_size / machine_size * 1
    arrival_time = torch.rand(size=((1,problem_size-4)))
    arrival_time, _ = torch.sort(arrival_time, dim=-1, descending=False)
    arrival_time = arrival_time * avg_proc.item()
    arrival_time[0,-1] = 99999

    #Step completion rate
    sc_rate = torch.range(start=0, end=(operation_size-1))/(operation_size-1)
    sc_rate = sc_rate[None,None,:,None].expand(batch_size, problem_size, operation_size, 1)

    # due time
    temp = torch.sum(torch.mean(proc,dim=-1),dim=-1)
    EL = torch.randint(1, 4, size = (batch_size, problem_size)) * 0.5 + 0.2
    due_time = temp * EL
    EL = EL[:,:,None,None].expand(batch_size, problem_size, operation_size, 1)/1.7

    #
    arr_time = torch.zeros(size=(batch_size, problem_size, operation_size, 1))

    #problems = torch.cat((proc[:,:,:,:], ec_proc[:,:,:,:], sc_rate, EL, arr_time), dim=-1)
    problems = torch.cat((proc[:, :, :, :], sc_rate, EL, arr_time), dim=-1)
    #temp = torch.ones(size=(batch_size, problem_size, 1, machine_size*2 + 3))
    temp = torch.ones(size=(batch_size, problem_size, 1, machine_size + 3))
    problems = torch.cat((problems, temp), dim=2)

    return problems, arrival_time, due_time


def get_random_problems_test(batch_size, problem_size, operation_size, machine_size):
    #proc = torch.rand(size=(batch_size, problem_size, operation_size, machine_size))

    proc = torch.randint(1, 100, size=(batch_size, problem_size, operation_size)).unsqueeze(3).repeat(1, 1, 1, machine_size)
    proc_rate = (torch.rand(size=(batch_size, problem_size, operation_size, machine_size)) * 4 / 10) + 0.8
    proc_out = torch.floor(proc_rate * proc).clamp_(min=1, max=99)
    proc = proc_out / 100

    avg_proc = torch.mean(proc) * problem_size * operation_size / machine_size * 1
    arrival_time = torch.rand(size=((1,problem_size-4)))
    arrival_time, _ = torch.sort(arrival_time, dim=-1, descending=False)
    arrival_time = arrival_time * avg_proc.item()
    arrival_time_out = torch.zeros(1, problem_size)
    arrival_time_out[:,5:problem_size] = arrival_time[:,:(problem_size-5)] * 100

    arrival_time[0,-1] = 99999

    #Step completion rate
    sc_rate = torch.range(start=0, end=(operation_size-1))/(operation_size-1)
    sc_rate = sc_rate[None,None,:,None].expand(batch_size, problem_size, operation_size, 1)

    # due time
    temp = torch.sum(torch.mean(proc,dim=-1),dim=-1)
    EL = torch.randint(1, 4, size = (batch_size, problem_size)) * 0.5 + 0.2
    EL_out = EL
    due_time = temp * EL
    due_time_out = (due_time * 100) + arrival_time_out
    EL = EL[:,:,None,None].expand(batch_size, problem_size, operation_size, 1)/1.7

    #
    arr_time = torch.zeros(size=(batch_size, problem_size, operation_size, 1))

    problems = torch.cat((proc[:,:,:,:], sc_rate, EL, arr_time), dim=-1)
    temp = torch.ones(size=(batch_size, problem_size, 1, machine_size + 3))
    problems = torch.cat((problems, temp), dim=2)
    #problems = torch.cat((temp, problems), dim=1)


    return problems, arrival_time, due_time, proc_out.reshape(-1,machine_size), arrival_time_out.squeeze(), EL_out.squeeze(), due_time_out.squeeze()