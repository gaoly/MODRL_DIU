import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KPModel(nn.Module):

    def __init__(self, PS, EC_PROC, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = KP_Encoder(**model_params)
        self.decoder = KP_Decoder(**model_params)
        self.encoded_nodes_prev = None
        self.encoded_nodes = None
        self.encoded_graph = None

        self.EC_PROC = EC_PROC
        self.PS = PS[:,0,0,:].unsqueeze(1).repeat(1,EC_PROC.size(1),1)
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state, step_state = None):
        # shape: (batch, pomo, problem, EMBEDDING_DIM)

        problems = reset_state.problems[:,:,0:step_state.current_jobnum,:,:]

        batch_size = problems.size(0)
        pomo_size = problems.size(1)
        problem_size = problems.size(2)
        course_size = problems.size(3)
        feature_size = problems.size(4)

        problems = problems.reshape(batch_size, pomo_size, course_size * problem_size, -1)
        problems = problems[step_state.op_idx[:,:,:,0:(int(step_state.op_idx.size(-1)/2))].reshape(batch_size, pomo_size, -1) == 1].reshape(batch_size * pomo_size, -1, feature_size)

        # 64 50 128
        self.encoded_nodes = self.encoder(problems)
        self.encoded_nodes = self.encoded_nodes.reshape(batch_size, pomo_size, -1, self.model_params['embedding_dim'])



    def pre_forward_period(self, step_state):
        # shape: (batch, problem, EMBEDDING_DIM)
        batch_size = self.encoded_nodes.size(0)
        pomo_size = self.encoded_nodes.size(1)
        feature_size = self.encoded_nodes.size(-1)
        course_idx = step_state.course_idx[:,:,:,:].reshape(batch_size, pomo_size, -1)

        course_idx = course_idx[step_state.op_idx[:,:,:,0:(int(step_state.op_idx.size(-1)/2))].reshape(batch_size, pomo_size, -1) == 1].reshape(batch_size , pomo_size, -1)
        self.encoded_nodes_prev = self.encoded_nodes[course_idx == 1].reshape(batch_size , pomo_size, -1, feature_size)
        # batch, POMO ,problem, EMBEDDING_DIM
        self.encoded_graph = self.encoded_nodes_prev.mean(dim=2, keepdim=True)
        self.decoder.set_kv(self.encoded_nodes_prev, self.normalize(step_state.machine_available_time), self.normalize(step_state.job_available_time[step_state.complete_job_mask == 1].reshape(batch_size, pomo_size, -1)), self.EC_PROC/torch.max(self.EC_PROC),self.PS/torch.max(self.PS))

    def normalize(self, reward):
        max_val = torch.max(reward,-1)[0]
        min_val = torch.min(reward, -1)[0]
        if ((max_val - min_val) == 0).any():
            reward_n = reward
        else:
            reward_n = (reward - min_val[:,:,None])/(max_val[:,:,None] - min_val[:,:,None])
        return reward_n
    def set_gru(self):
        self.decoder.set_gru(self.encoded_graph)

    def set_gru_period(self, step_state):
        selected_encoded_nodes_prev = self.encoded_nodes_prev[step_state.BATCH_IDX, step_state.POMO_IDX, step_state.selected_job_fake]
        self.decoder.set_gru_period(step_state, selected_encoded_nodes_prev)

        # batch, POMO ,problem, EMBEDDING_DIM

    def forward(self, state, reset_state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        #shape: (batch, pomo, embedding)
        probs = self.decoder(self.encoded_graph, machine_available_time = state.machine_available_time, ninf_mask=state.ninf_mask)

        # shape: (batch, pomo, problem)
        if self.training or self.model_params['eval_type'] == 'softmax':
            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                .squeeze(dim=1).reshape(batch_size, pomo_size)
            # shape: (batch, pomo)

            prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                .reshape(batch_size, pomo_size)

        else:
            selected = probs.argmax(dim=2)
            # shape: (batch, pomo)
            prob = None
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class KP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        #self.embedding = nn.Linear(2, embedding_dim)
        self.embedding = nn.Linear(self.model_params['feature_size'], self.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################
class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.machine_size = self.model_params['machine_size']

        #GRU
        num_layers = 1
        self.GRUs = nn.GRU(embedding_dim, embedding_dim, num_layers,
                              batch_first=True,
                              dropout=0.1 if num_layers > 1 else 0)
        self.last_hh = None
        self.gru_out = None
        self.hyper_gru = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.hyper_gru_in = nn.Parameter(torch.zeros((1, self.machine_size, embedding_dim, embedding_dim), requires_grad=True))
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        hyper_input_dim = 8
        hyper_hidden_embd_dim = 256
        self.embd_dim = 8
        self.hyper_output_dim = 5 * self.embd_dim

        self.hyper_fc0 = nn.Linear(embedding_dim, embedding_dim * self.machine_size, bias=True)

        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)
        
        self.hyper_Wq = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, (4 + embedding_dim) * head_num * qkv_dim * 2, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, (4 + embedding_dim) * head_num * qkv_dim * 2, bias=False)
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * 2 * (embedding_dim + 4), bias=False)
        self.hyper_Wq_gru = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)

        self.Wq_para = None
        self.multi_head_combine_para = None
        
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention


    def assign(self, pref, step_state, step):
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        batch_size = self.model_params['train_batch_size']
        pomo_size = self.model_params['pomo_size']

        selected_size = step_state.current_jobnum * self.model_params['operation_size']
        temp = torch.arange(selected_size + 1)[:, None]
        p_rate = torch.cat((temp, -temp + selected_size), dim=1) / selected_size

        pref = torch.cat((pref, p_rate[step].squeeze()), dim=0)
        pref = pref[None, None, :].expand(step_state.dynamic_global_info.shape[0], pomo_size, 5)
        pref = torch.cat((pref, step_state.dynamic_global_info), dim=-1)

        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)

        self.Wq_para = self.hyper_Wq(mid_embd[:,:,:self.embd_dim]).reshape(batch_size, pomo_size, head_num * qkv_dim, embedding_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[:,:,1 * self.embd_dim: 2 * self.embd_dim]).reshape(batch_size, pomo_size,head_num * qkv_dim * 2, (embedding_dim+4))
        self.Wv_para = self.hyper_Wv(mid_embd[:,:,2 * self.embd_dim: 3 * self.embd_dim]).reshape(batch_size, pomo_size,head_num * qkv_dim * 2, (embedding_dim+4))
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[:,:,3 * self.embd_dim: 4 * self.embd_dim]).reshape(batch_size, pomo_size,(embedding_dim+4), head_num * qkv_dim * 2)
        self.Wq_gru_para = self.hyper_Wq_gru(mid_embd[:,:,4 * self.embd_dim: 5 * self.embd_dim]).reshape(
            batch_size, pomo_size, head_num * qkv_dim, embedding_dim)

    def set_kv(self, encoded_nodes, machine_available_time, job_available_time, EC_PROC, PS):
        #encoded_nodes batch, POMO ,problem, EMBEDDING_DIM
        #machine_available_time batch, POMO, machine_size
        #job_available_time batch, POMO, job_size

        batch_size = encoded_nodes.size(0)
        head_num = self.model_params['head_num']
        problem_size = encoded_nodes.size(2)
        machine_num = machine_available_time.size(2)
        pomo_size = machine_available_time.size(1)
        # hyper_encoded_nodes

        hyper_encoded_nodes = self.hyper_fc0(encoded_nodes).reshape(batch_size, pomo_size, problem_size * machine_num, -1)
        job_available_time = job_available_time.repeat_interleave(repeats=machine_num, dim=2)

        machine_available_time = machine_available_time[:,:,None,:].repeat_interleave(repeats=problem_size, dim=2)
        machine_available_time = machine_available_time.reshape(batch_size, pomo_size, -1)

        EC_PROC = EC_PROC[:, :, None, :].repeat_interleave(repeats=problem_size, dim=2)
        EC_PROC = EC_PROC.reshape(batch_size, pomo_size, -1)

        PS = PS[:, :, None, :].repeat_interleave(repeats=problem_size, dim=2)
        PS = PS.reshape(batch_size, pomo_size, -1)


        encoded_nodes = torch.cat((hyper_encoded_nodes, job_available_time[:,:,:,None], machine_available_time[:,:,:,None], EC_PROC[:,:,:,None], PS[:,:,:,None]), dim=3)


        self.k = reshape_by_heads_5D(torch.matmul(encoded_nodes, self.Wk_para.transpose(-1, -2)), head_num=head_num)
        self.v = reshape_by_heads_5D(torch.matmul(encoded_nodes, self.Wv_para.transpose(-1, -2)), head_num=head_num)

        self.single_head_key = encoded_nodes.transpose(2, 3)

    def set_gru(self, encoded_graph):
        batch_size = encoded_graph.size(0)
        pomo_size = encoded_graph.size(1)
        embedding_dim = self.model_params['embedding_dim']
        machine_size = self.machine_size
        head_num = self.model_params['head_num']

        encoded_graph = encoded_graph.reshape(batch_size, pomo_size, embedding_dim)
        gru_out, last_hh = self.GRUs(encoded_graph, None)
        # batch_size * pomo_size, machine_size, embedding_dim
        self.last_hh = last_hh

        self.gru_embedding = self.hyper_gru(gru_out.reshape(batch_size, pomo_size, embedding_dim))

        self.gru_embedding = reshape_by_heads(torch.matmul(self.gru_embedding[:,:,None,:], self.Wq_gru_para.transpose(-1, -2)).squeeze(dim=2), head_num=head_num)
        #self.gru_embedding = reshape_by_heads(F.linear(self.gru_embedding, self.Wq_gru_para), head_num=head_num)
    def set_gru_period(self, step_state, selected_encoded_nodes_prev):
        batch_size = selected_encoded_nodes_prev.size(0)
        pomo_size = selected_encoded_nodes_prev.size(1)
        embedding_dim = self.model_params['embedding_dim']
        machine_size = self.machine_size
        head_num = self.model_params['head_num']

        hyper_gru_in = self.hyper_gru_in.repeat_interleave(repeats=batch_size * pomo_size, dim=0).reshape(batch_size, pomo_size, machine_size, embedding_dim, embedding_dim)
        hyper_gru_in = hyper_gru_in[step_state.BATCH_IDX, step_state.POMO_IDX, step_state.selected_machine]

        last_hh = self.last_hh
        selected_encoded_nodes_prev = selected_encoded_nodes_prev.reshape(batch_size, pomo_size, embedding_dim)
        selected_encoded_nodes_prev = torch.matmul(selected_encoded_nodes_prev[:,:,None,:], hyper_gru_in)
        gru_out, last_hh = self.GRUs(selected_encoded_nodes_prev.squeeze(dim=2), last_hh)

        self.last_hh = last_hh

        self.gru_embedding = self.hyper_gru(gru_out.reshape(batch_size, pomo_size, embedding_dim))
        self.gru_embedding = reshape_by_heads(torch.matmul(self.gru_embedding[:, :, None, :], self.Wq_gru_para.transpose(-1, -2)).squeeze(dim=2), head_num=head_num)
        #self.gru_embedding = reshape_by_heads(F.linear(self.gru_embedding, self.Wq_gru_para), head_num=head_num)
    def forward(self, graph, machine_available_time, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        batch_size = machine_available_time.size(0)
        group_size = machine_available_time.size(1)
        machine_num = machine_available_time.size(2)

        #  Multi-Head Attention
        #######################################################
        input_cat = graph.squeeze(dim=2)

        q = reshape_by_heads(torch.matmul(input_cat[:,:,None,:], self.Wq_para.transpose(-1, -2)).squeeze(dim=2), head_num=head_num)
        q = torch.cat((q, self.gru_embedding), dim=3)

        out_concat = multi_head_attention_de(q, self.k, self.v, ninf_mask=ninf_mask)

        mh_atten_out = torch.matmul(out_concat[:,:,None,:], self.multi_head_combine_para.transpose(-1, -2)).squeeze(dim=2)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out[:,:,None,:], self.single_head_key).squeeze(dim=2)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        #score_masked = score_clipped + ninf_mask
        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)
        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def reshape_by_heads_5D(qkv, head_num):
    # q.shape: (batch, pomo, pomo*machine_num, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)
    D3 = qkv.size(2)

    q_reshaped = qkv.reshape(batch_s, n, D3, head_num, -1)
    # shape: (batch, n, D3, head_num, key_dim)

    q_transposed = q_reshaped.transpose(2, 3)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, TSP_SIZE)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat

def multi_head_attention_de(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = q.size(2)
    q = q.transpose(1, 2)[:,:,:,None,:]

    score = torch.matmul(q, k.transpose(3, 4))
    # shape = (batch, head_num, n, TSP_SIZE)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, :, None, None, :].expand(batch_s, n, head_num, 1, score_scaled.size(4))

    weights = nn.Softmax(dim=4)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)

    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(2, 3)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))