Attr Rollout 
===
Official implementation of Attribution Rollout: A New Way to Interpret Visual Transformer.
We provide a [jupyter notebook](examole.ipynb) for quickly experience the visualization of our approach

# Credits
ViT implementation is based on:
- https://github.com/hila-chefer/Transformer-Explainability
- https://github.com/rwightman/pytorch-image-models
- https://github.com/lucidrains/vit-pytorch
- https://github.com/XianrenYty/Transition_Attention_Maps

Evaluation experiments is based on:
- https://github.com/hila-chefer/Transformer-Explainability
- 
# Reproducing evaluation results

## Perturbation Test
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/generate_visualizations.py --method attr_rollout --imagenet-validation-path path/to/imagenet/'ImageNet 2012 DataSets'
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/pertubation_eval_from_hdf5.py --method attr_rollout
```

##Segmentation Results

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/imagenet_seg_eval.py --method attr_rollout_max --imagenet-seg-path /path/to/gtsegs_ijcv.mat
```


