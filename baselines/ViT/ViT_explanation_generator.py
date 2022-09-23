import argparse
import torch
import numpy as np
from numpy import *
import torch.nn as nn

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]
    
    def generate_transition_attention_maps(self, input, index=None, start_layer=0, steps=20, with_integral=True, first_state=False):
        b = input.shape[0]
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            states = states.bmm(attn)
            # add residual
            states += states_

        total_gradients = torch.zeros(b, h, s, s).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients
        
        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        states = states * W_state
        
        return states[:, 0, 1:]
    
    def generate_hao(self, input, index=None, start_layer=0, steps=20, with_integral=False, first_state=False):
        b = input.shape[0]
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            states = states.bmm(attn)
            # add residual
            states += states_

        total_gradients = torch.zeros(b, h, s, s).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients
        
        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        states = states * W_state
        
        return states[:, 0, 1:]

class IG:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_ig(self, input):
        output = self.model(input.cuda(), register_hook=True, retain_graph=True)
        blocks = self.model.blocks
        all_layer_attentions = []
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        heads = 12
        m = 20
        steps = 20
        scale = 1.0 / m
        prob = self.model.get_ig()
        k=0
        discard_ratio = 0.9
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            F = prob[k]
            k = k + 1
            all_ig = []
            for i in range(heads):
                attn = attn_heads[0][i]
                gradient = torch.autograd.grad(torch.unbind(F[:, index]), attn_heads, retain_graph=True)
                gradient = gradient[0][0][i]
                baseline = torch.zeros_like(gradient)
                for alpha in np.linspace(0,1,steps):
                    gradient_a = gradient * alpha
                    baseline += gradient_a
                attn = attn * baseline * scale
                all_ig.append(attn)
            a_ig = torch.stack(all_ig,dim=0)
            a_ig = a_ig.unsqueeze(dim=0)
            ig = torch.max(a_ig,dim=1)[0]
            all_layer_attentions.append(ig)
        return all_layer_attentions            