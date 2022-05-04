import os
from tqdm import tqdm
import h5py

import argparse

# Import saliency methods and models
from misc_functions import *

from ViT_explanation_generator import Baselines, LRP, IG
from ViT_new import vit_base_patch16_224
from ViT_LRP import vit_base_patch16_224 as vit_LRP
from ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP
from ViT_ig import vit_base_patch16_224 as vit_attr_rollout

from torchvision.datasets import ImageNet

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

def compute_rollout_max_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [(all_layer_matrices[i] + 1.0 * eye)/2 for i in range(len(all_layer_matrices))]
    
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def compute_rollout_min_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [(all_layer_matrices[i].min(axis=0)[0] + 1.0 * eye)/2 for i in range(len(all_layer_matrices))]
    
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(args):
    first = True
    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            data.requires_grad_()

            index = None
            if args.vis_class == 'target':
                index = target

            if args.method == 'rollout':
                Res = baselines.generate_rollout(data, start_layer=1).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'tam':
                Res = baselines.generate_transition_attention_maps(data, index=index, start_layer=4).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'hao':
                Res = baselines.generate_hao(data, index=index, start_layer=4).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'lrp':
                Res = lrp.generate_LRP(data, start_layer=1, index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'transformer_attribution':
                Res = lrp.generate_LRP(data, start_layer=1, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'full_lrp':
                Res = orig_lrp.generate_LRP(data, method="full", index=index).reshape(data.shape[0], 1, 224, 224)
                # Res = Res - Res.mean()

            elif args.method == 'lrp_last_layer':
                Res = orig_lrp.generate_LRP(data, method="last_layer", is_ablation=args.is_ablation, index=index) \
                    .reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'attn_last_layer':
                Res = lrp.generate_LRP(data, method="last_layer_attn", is_ablation=args.is_ablation) \
                    .reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'attn_gradcam':
                Res = baselines.generate_cam_attn(data, index=index).reshape(data.shape[0], 1, 14, 14)
                
            elif args.method == 'attr_rollout':
                Res = ig.generate_ig(data)
                Res = compute_rollout_attention(Res)
                Res = Res[:,0, 1:]
                Res = Res.reshape(data.shape[0], 1, 14, 14)
                
            elif args.method == 'attr_rollout_max':
                Res = ig.generate_ig(data)
                Res = compute_rollout_max_attention(Res)
                Res = Res[:,0, 1:]
                Res = Res.reshape(data.shape[0], 1, 14, 14)
                
            elif args.method == 'attr_rollout_min':
                Res = ig.generate_ig(data)
                Res = compute_rollout_min_attention(Res)
                Res = Res[:,0, 1:]
                Res = Res.reshape(data.shape[0], 1, 14, 14)
                
            if args.method != 'full_lrp' and args.method != 'input_grads':
                Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam','tam', 'attr_rollout','attr_rollout_max','attr_rollout_min','hao'],
                        help='')
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--imagenet-validation-path', type=str,
                        required=True,
                        help='')
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    model = vit_base_patch16_224(pretrained=True).cuda()
    baselines = Baselines(model)

    # LRP
    model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)

    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)
    
    # attr_rollout
    model_attr_rollout = vit_attr_rollout(pretrained=True).cuda()
    model_attr_rollout.eval()
    ig = IG(model_attr_rollout)

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    compute_saliency_and_save(args)
