import argparse
import torch
import numpy as np
from numpy import *

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


class Deeplift:
    def __init__(self,model):
        self.model = model
        self.model.eval()

    def _setup(self):
        self._prepare_layers()

        self._replace_forward_method()
        self._register_backward_hook()

    def _register_backward_hook(self):
        device = self.device

        def register_hook(layer):
            def rescale_hook(self, input, grad_out):
                # same threshold is used in original implementation
                # https://github.com/kundajelab/deeplift/blob/7cb4804cc8e682662652ae24903d9492b4d74523/deeplift/util.py#L14
                # Data points with values lower than the threshold do not use the multiplier, but the gradient directly.
                near_zero_threshold = 1e-7
                alpha = 0.01

                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)

                ref_y = self.outputs[0].to(device)
                y = self.outputs[1].to(device)
                delta_y = (y - ref_y).to(device)

                multiplier = (delta_y / (delta_x + 1e-7)).to(device)

                far_zero_contrib_mask = (delta_x.abs() > near_zero_threshold).float().to(device)
                with_multiplier = far_zero_contrib_mask * multiplier
                with_multiplier = with_multiplier.to(device)

                near_zero_contrib_mask = (delta_x.abs() <= near_zero_threshold).float().to(device)
                without_multiplier = near_zero_contrib_mask * alpha
                without_multiplier = without_multiplier.to(device)

                scale_factor = with_multiplier + without_multiplier
                output = (scale_factor * input[0]).to(device)

                return (output,)

            def reveal_cancel_hook(self, input, grad_out):
                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)
                delta_x_plus = ((delta_x >= 0).float().to(device))*delta_x
                delta_x_minus = ((delta_x < 0).float().to(device))*delta_x

                delta_y_plus = 0.5*(F.relu(ref_x + delta_x_plus) - F.relu(ref_x)) +\
                               0.5*(F.relu(ref_x + delta_x_plus + delta_x_minus) - F.relu(ref_x + delta_x_minus))

                delta_y_minus = 0.5 * (F.relu(ref_x + delta_x_minus) - F.relu(ref_x)) + \
                                0.5 * (F.relu(ref_x + delta_x_plus + delta_x_minus) - F.relu(ref_x + delta_x_plus))

                m_x_plus = delta_y_plus / (delta_x_plus + 1e-7)
                m_x_plus *= 1
                m_x_minus = delta_x_minus / (delta_y_minus + 1e-7)
                m_x_minus *= 1

                grad = input[0]
                grad_plus = ((grad >= 0).float().to(device)) * grad
                grad_minus = ((grad < 0).float().to(device)) * grad

                output = grad_plus * m_x_plus + grad_minus * m_x_minus

                return (output,)

            def linear_hook(self, input, grad_out):
                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)
                delta_x_plus = ((delta_x > 0).float().to(device))
                delta_x_minus = ((delta_x < 0).float().to(device))
                delta_x_zero = ((delta_x == 0).float().to(device))

                transposed_weight = self.weight.detach().clone().T.to(device)
                size = transposed_weight.size()
                transpose_pos = nn.Linear(size[1], size[0]).to(device)
                transpose_pos.weight = nn.Parameter(((transposed_weight > 0).float().to(device)) * transposed_weight)
                transpose_negative = nn.Linear(size[1], size[0]).to(device)
                transpose_negative.weight = nn.Parameter(((transposed_weight < 0).float().to(device)) * transposed_weight)

                transpose_full = nn.Linear(size[1], size[0]).to(device)
                transpose_full.weight = nn.Parameter(transposed_weight)

                ref_y = self.outputs[0].to(device)
                y = self.outputs[1].to(device)
                delta_y = (y - ref_y).to(device)
                delta_y_plus = ((delta_y > 0).float().to(device)) * delta_y
                delta_y_minus = ((delta_y < 0).float().to(device)) * delta_y

                pos_grad_out = delta_y_plus * grad_out[0]
                neg_grad_out = delta_y_minus * grad_out[0]

                pos_pos_result = transpose_pos.forward(pos_grad_out) * delta_x_plus
                pos_neg_result = transpose_pos.forward(neg_grad_out) * delta_x_plus
                neg_pos_result = transpose_negative.forward(neg_grad_out) * delta_x_minus
                neg_neg_result = transpose_negative.forward(pos_grad_out) * delta_x_minus
                null_result = transpose_full.forward(grad_out[0]) * delta_x_zero

                multiplier = pos_pos_result + pos_neg_result + neg_pos_result + neg_neg_result + null_result

                out = (input[0],) + (multiplier.to(device),) + input[2:]
                return out

            def linear_conv_hook(self, input, grad_out):
                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)
                delta_x_plus = ((delta_x > 0).float().to(device))
                delta_x_minus = ((delta_x < 0).float().to(device))
                delta_x_zero = ((delta_x == 0).float().to(device))

                transpose_pos = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding).to(device)
                transpose_pos.weight = nn.Parameter(((self.weight > 0).float().to(device))*self.weight.detach().clone().to(device)).to(device)

                transpose_negative = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding).to(device)
                transpose_negative.weight = nn.Parameter(((self.weight < 0).float().to(device)) * self.weight.detach().clone()).to(device)

                transpose_full = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size,
                                                        self.stride, self.padding).to(device)
                transpose_full.weight = nn.Parameter((self.weight.detach().clone().to(device))).to(device)

                ref_y = self.outputs[0].to(device)
                y = self.outputs[1].to(device)
                delta_y = (y - ref_y).to(device)
                delta_y_plus = ((delta_y > 0).float().to(device)) * delta_y
                delta_y_minus = ((delta_y < 0).float().to(device)) * delta_y

                pos_grad_out = delta_y_plus * grad_out[0]
                neg_grad_out = delta_y_minus * grad_out[0]

                dim_check = transpose_pos.forward(pos_grad_out)
                if dim_check.shape != delta_x.shape:
                    if dim_check.shape[3] > delta_x.shape[3]:
                        dim_diff = dim_check.shape[3] - delta_x.shape[3]
                        delta_x = torch.cat((delta_x, torch.ones(delta_x.shape[0], delta_x.shape[1], dim_diff, delta_x.shape[3])), 2)
                        delta_x = torch.cat((delta_x, torch.ones(delta_x.shape[0], delta_x.shape[1], delta_x.shape[2], dim_diff)), 3)
                    else:
                        new_shape = dim_check.shape
                        delta_x = delta_x[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]

                    delta_x_plus = ((delta_x > 0).float().to(device))
                    delta_x_minus = ((delta_x < 0).float().to(device))
                    delta_x_zero = ((delta_x == 0).float().to(device))

                pos_pos_result = transpose_pos.forward(pos_grad_out) * delta_x_plus
                pos_neg_result = transpose_pos.forward(neg_grad_out) * delta_x_plus
                neg_pos_result = transpose_negative.forward(neg_grad_out) * delta_x_minus
                neg_neg_result = transpose_negative.forward(pos_grad_out) * delta_x_minus
                null_result = transpose_full.forward(grad_out[0]) * delta_x_zero

                multiplier = pos_pos_result + pos_neg_result + neg_pos_result + neg_neg_result + null_result

                if input[0].shape != multiplier.shape:
                    if input[0].shape[3] > multiplier.shape[3]:
                        dim_diff = input[0].shape[3] - multiplier.shape[3]
                        multiplier = torch.cat((multiplier, torch.ones(multiplier.shape[0], multiplier.shape[1], dim_diff, multiplier.shape[3])), 2)
                        multiplier = torch.cat((multiplier, torch.ones(multiplier.shape[0], multiplier.shape[1], multiplier.shape[2], dim_diff)), 3)
                    else:
                        new_shape = input[0].shape
                        multiplier = delta_x[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]

                out = (multiplier.to(device),) + input[1:]
                return out

            if self.rule != DeepLIFTRules.NoRule:
                if isinstance(layer, torch.nn.ReLU) and self.rule != DeepLIFTRules.Linear:
                    if self.rule == DeepLIFTRules.LinearRescale or self.rule == DeepLIFTRules.Rescale:
                        self.hooks.append(layer.register_backward_hook(rescale_hook))
                    else:
                        self.hooks.append(layer.register_backward_hook(reveal_cancel_hook))
                elif isinstance(layer, torch.nn.Linear) and \
                        (self.rule == DeepLIFTRules.LinearRescale or self.rule == DeepLIFTRules.LinearRevealCancel or self.rule == DeepLIFTRules.Linear):
                    self.hooks.append(layer.register_backward_hook(linear_hook))
                elif isinstance(layer, torch.nn.Conv2d) and \
                        (self.rule == DeepLIFTRules.LinearRescale or self.rule == DeepLIFTRules.LinearRevealCancel or self.rule == DeepLIFTRules.Linear):
                    self.hooks.append(layer.register_backward_hook(linear_conv_hook))
            self.model.apply(register_hook)

    def _replace_forward_method(self):
        device = self.device

        def add_forward_hook(layer):
            def forward_hook(self, input, output):
                self.inputs.append(input[0].data.clone().to(device))
                self.outputs.append(output[0].data.clone().to(device))

            if self.rule != DeepLIFTRules.NoRule:
                self.hooks.append(layer.register_forward_hook(forward_hook))

        self.model.apply(add_forward_hook)

    def _prepare_layers(self):
        def _init_layers(layer):
            layer.inputs = []
            layer.outputs = []

        self.model.apply(_init_layers)


    def generate_DeepLift(self,input):
        self._setup()
        output = self.model(input)
        grad_out = torch.zeros(output.shape).to(self.device)
        grad_out[0][self._last_prediction] = 1.0
        output.backward(grad_out)
        deeplift = (output - output.min()) / (output.max() - output.min()) 
        return deeplift


        

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
