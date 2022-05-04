#Attr Rollout 
Official implementation of Attribution Rollout: A New Way to Interpret Visual Transformer.
We provide a [jupyter notebook](examole.ipynb) for quickly experience the visualization of our approach

#Reproducing evaluation results

##Perturbation Test

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/generate_visualizations.py --method attr_rollout --imagenet-validation-path path/to/imagenet/'ImageNet 2012 DataSets'
