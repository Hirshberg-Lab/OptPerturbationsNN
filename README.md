# First-Passage Approach to Optimizing Perturbations for Improved Training of Machine Learning Models
To use this code please install the ffcv library

To run the code for the shrink & perturb protocol write:
```python
python create_acc_dist.py --use_ffcv True --lr [choose learning rate] --momentum [choose momentum] --reset_rate [choose perturbation interval] --model_name 'resnet18' --shrink 0.4 --perturb 0.1 --n_epochs [choose number of epochs] --n_samples [choose number of samples] --url [choose directory to save (it need to be a string)]+'/testacc_list_s_and_p_reset7/'
```
To run the code for the partial resetting protocol write:
```python
python create_acc_dist.py --use_ffcv True --lr [choose learning rate] --momentum [choose momentum] --reset_rate [choose perturbation interval] --model_name 'resnet18' --n_epochs [choose number of epochs] --n_samples [choose number of samples] --url [choose directory to save (it need to be a string)]+'/testacc_list_partial_reset/' --perturbation_kind '_perform_reset_small_weights' --fraction [choose fraction of weights to re-initialize (float)]
```
To run regular training write:
```python
python create_acc_dist.py --use_ffcv True --lr [choose learning rate] --momentum [choose momentum] --model_name 'resnet18'  --n_epochs 200 --n_samples 1000 --url [choose directory to save (it need to be a string)]+'/testacc_list_reg_training/'
```