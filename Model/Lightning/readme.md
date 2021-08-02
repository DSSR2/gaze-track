## PyTorch Lightning Training 
### Usage: 
```
python lit_train.py 
    --dataset_dir <Path to dataset> 
    --save_dir <Path to save files> 
    --gpus <Number of GPUs to use> 
    --epochs <Number of epochs>
    --comet_name <Name of the experiment on comet.ml>
    --batch_size <Batch size>
    --checkpoint <Path to load checkpoint from to continue training>

```

### Example: 
```
python lit_train.py --dataset_dir ../gazetrack/ --save_dir ../Checkpoints/ --gpus 1 --epochs 50 --checkpoint ../Checkpoints/checkpoint.ckpt
```