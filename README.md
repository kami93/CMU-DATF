# Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding

This code is PyTorch implementation of [Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding](https://arxiv.org/abs/2003.03212).

![Model Diagram](figures/model_figure.png)

## Requirements
Python 3 and the following external packages.
- PyTorch>=1.7
- torchvision
- scipy
- numpy
- openCV
- pyquaternion
- shapely
- pykalman
- [Argoverse API](https://github.com/argoai/argoverse-api)
- [nuScenes API](https://github.com/nutonomy/nuscenes-devkit)

## Downloading Dataset

You will have to download [Argoverse](https://www.argoverse.org/data.html#download-link) and [nuScenes](https://www.nuscenes.org/download) dataset into **data/Argoverse** and  **data/nuScenes**, then verify the subdirectory of each dataset. Dataset should have structure same as:

```
-Argoverse
  |- test_obs/data/*.csv
  |- train/data/*.csv
  |- val/data/*.csv

-nuScenes
  |- sweeps/"Sensor Name (e.g., LIDAR_TOP)"/*
  |- trainval_meta
      |- maps/*.json
      |- v1.0-trainval/*.json
```


## Preprocessing Dataset

Both Argoverse and nuScenes dataset have to be preprocessed prior to testing this model. To do so, you may simply run python scripts **preprocess_argoverse.py** and **preprocess_nuscenes.py**. Each script will generate distance transform of drivable area map and partition **train, train_val,** and **val** (plus, **test_obs** for Argoverse) splits for training. Note that these names do not represent ***train/validation/test*** regime in general machine learning literature. Instead, they are derived from the naming done by authors for each dataset. In this model, overall, the **train** split corresponds to the ***train set***, the **train_val** split corresponds to the ***validation set***, and the **val** split corresponds to the ***test set***. The scripts will additionally generate symlinks to form **train_all** split, which is the union of **train** and **train_val** to be used for finetuning this model.

In addition, **preprocess_nuscenes.py** will perform Kalman smoothing with constant velocity model to generate trajectories from tracklets since the original nuScenes dataset is not geared to trajectories task (UPDATE: nuScenes now has the [*prediction* task](https://www.nuscenes.org/prediction) open).

The preprocessed datasets will be saved at **data/Preprocessed/**.


## Initial Training
This model is initially trained using **train** split with learning rate decay on plateau depending on performances measured using **train_val** split. See comments and values in main.py for the hyperparameter details. To train the proposed method under a minimal set of options, run the following commands.

**nuScenes**
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=nuscenes \
--train_partition=train --train_cache=./train.nusc.pkl \
--val_partition=train_val --val_cache=./train_val.nusc.pkl \
--tag=attgscam.nusc --batch_size=8 
```
**Argoverse**
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=argoverse \
--train_partition=train --train_cache=./train.argo.pkl \
--val_partition=train_val --val_cache=./train_val.argo.pkl \
--tag=attgscam.argo --batch_size=8 
```

## Fine-tuning
After the initial training is done either due to reaching the maximum set epochs or the maximum set learning rate decays, one must perform finetuning to achieve the best performance in testing time. To do so, set the training split to ***train_all*** and provide the location of the initial training log and the restoring epoch to arguments *--restore_path* and *--restore_epoch*. In addition, set a small learning rate value (e.g., 1e-5) which should be much smaller than that used during the initial training. Also, the learning rate decay on plateau must be turned off or the model would cheat by adapting its training schedule to achieve a good performance on ***val*** split. Usually, an epoch of fine-tuning suffices to achieve the best performance; fine-tuning for more than two epochs may harm the performance.

To fine-tune the proposed method under a minimal set of options, run the following commands.

**nuScenes**
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=nuscenes \
--train_partition=train_all --train_cache=./train_all.nusc.pkl \
--val_partition=val --val_cache=./val.nusc.pkl \
--tag=attgscam.finetune.nusc --batch_size=8 --init_lr=1e-5 --lr_decay=0 \
--restore_path=./experiment/attgscam.nusc* --restore_epoch=*
```

**Argoverse**
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=argoverse \
--train_partition=train_all --train_cache=./train_all.argo.pkl \
--val_partition=val --val_cache=./val.argo.pkl \
--tag=attgscam.finetune.argo --batch_size=8 --init_lr=1e-5 --lr_decay=0 \
--restore_path=./experiment/attgscam.argo* --restore_epoch=*
```

## Testing

Testing can be performed by assigning checkpoint to argument *--test_ckpt*. In the default setting, the test will output the mean and the standard deviation for ADEs/FDEs/rF/DAC/DAO after 10 testing trials on **val** split.

**nuScenes**
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=nuscenes \
--test_partition=val --test_cache=./val.nusc.pkl \
--tag=attgscam.finetune.nusc --batch_size=8 \
--test_ckpt=./experiment/attgscam.finetune.nusc*/*.pth.tar
```

**Argoverse**
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=argoverse \
--train_partition=train_all --train_cache=./train_all.argo.pkl \
--test_partition=val --test_cache=./val.argo.pkl \
--tag=attgscam.finetune.argo --batch_size=8 \
--test_ckpt=./experiment/attgscam.finetune.argo*/*.pth.tar
```

## Example Outputs

### Initial Training
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=nuscenes \
--train_partition=train --train_cache=./train.nusc.pkl \
--val_partition=train_val --val_cache=./train_val.nusc.pkl \
--tag=attgscam.nusc --batch_size=8

Experiment Path experiment/attgscam.nusc__31_March__22_17_.
Set random seed 88245.
.
.
.
Loading nuScenes dataset.
Found train set cache train.nusc.pkl with 15691 samples.
Data Loading Complete!
Found train_val set cache train_val.nusc.pkl with 6139 samples.
Data Loading Complete!
Train Examples: 15691
Valid Examples: 6139
Trainer Initialized!

Init LR: 1.000e-4, ReduceLROnPlateau: On
Decay patience: 5, Decay factor: 0.50, Num decays: 3
TRAINING START .....
==========================================================================================
------------------------------------------------------------------------------------------
| Epoch: 00 | Train Loss: 13.803186 | Train minADE[2/3]: 1.4996 / 2.4273 | Train minFDE[2/3]: 2.3533 / 4.2484 | Train avgADE[2/3]: 3.5996 / 5.3038 | Train avgFDE[2/3]: 6.0856 / 9.5970
| Epoch: 00 | Valid Loss: 8.407707 | Valid minADE[2/3]: 0.8077 / 1.2822 | Valid minFDE[2/3]: 1.1757 /2.1056 | Valid avgADE[2/3]: 2.0120 / 2.9221 | Valid avgFDE[2/3]: 3.3554 /5.2128 | Scheduler Metric: 3.3879 | Learning Rate: 1.000e-4
.
.
.
==========================================================================================
------------------------------------------------------------------------------------------
| Epoch: 66 | Train Loss: -7.240098 | Train minADE[2/3]: 0.3152 / 0.5650 | Train minFDE[2/3]: 0.5148 / 1.0795 | Train avgADE[2/3]: 0.6551 / 1.0646 | Train avgFDE[2/3]: 1.1580 / 2.1559
| Epoch: 66 | Valid Loss: 28.089247 | Valid minADE[2/3]: 0.3160 / 0.5695 | Valid minFDE[2/3]: 0.5239 /1.0990 | Valid avgADE[2/3]: 0.6499 / 1.0646 | Valid avgFDE[2/3]: 1.1571 /2.1697 | Scheduler Metric: 1.6685 | Learning Rate: 0.062e-4

Halt training since the lr decayed below 1.25e-05.
Training Complete!
```

### Finetuning
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=nuscenes --
train_partition=train_all --train_cache=./train_all.nusc.pkl --val_partition=val --val_cache=./val.nusc.pkl --tag=attgscam.finetune.nusc
--batch_size=8 --init_lr=1e-5 --lr_decay=0 --restore_path=./experiment/attgscam.nusc__31_March__22_17_/ --restore_epoch=66

Experiment Path experiment/attgscam.finetune.nusc__01_April__17_03_.
Set random seed 88245.
.
.
.
Loading nuScenes dataset.
Found train_all set cache train_all.nusc.pkl with 21830 samples.
Data Loading Complete!
Found val set cache val.nusc.pkl with 4669 samples.
Data Loading Complete!
Train Examples: 21830
Valid Examples: 4669
Loading checkpoint from experiment/attgscam.nusc__31_March__22_17_/ck_66_22.9269_51.6231_0.5695_1.0990.pth.tar
Trainer Initialized!

Init LR: 0.100e-4, ReduceLROnPlateau: Off
TRAINING START .....
==========================================================================================
------------------------------------------------------------------------------------------
| Epoch: 67 | Train Loss: -1.602181 | Train minADE[2/3]: 0.3196 / 0.5684 | Train minFDE[2/3]: 0.5112 / 1.0631 | Train avgADE[2/3]: 0.6848 / 1.1082 | Train avgFDE[2/3]: 1.2039 / 2.2368
| Epoch: 67 | Valid Loss: 6.696390 | Valid minADE[2/3]: 0.3364 / 0.6064 | Valid minFDE[2/3]: 0.5441 /1.1487 | Valid avgADE[2/3]: 0.7248 / 1.1825 | Valid avgFDE[2/3]: 1.2856 /2.4029 | Scheduler Metric: 1.7551 | Learning Rate: 0.100e-4

KeyboardInterrupt
```

### Testing
```
python main.py --model_type=AttGlobal_Scene_CAM_NFDecoder --dataset=nuscenes \
--test_partition=val --test_cache=./val.nusc.pkl \
--tag=attgscam.finetune.nusc --batch_size=8 \
--test_ckpt=./experiment/attgscam.finetune.nusc__01_April__17_03_/ck_67_1.4858_52.1059_0.6064_1.1487.pth.tar

Test Path tests/attgscam.finetune.nusc__01_April__17_41_.
Set random seed 88245.
.
.
.
Loading nuScenes dataset.
Found val set cache val.nusc.pkl with 4669 samples.
Data Loading Complete!
Test Examples: 4669
Tester Initialized!

Model Type: AttGlobal_Scene_CAM_NFDecoder
Velocity Const.: 0.50, Detach_Output: Off
Batchsize: 8
TESTING START .....
==========================================================================================
Working on test 1/10, batch 584/584... Complete.
minADE3: 0.609, minFDE3: 1.154
avgADE3: 1.186, avgFDE3: 2.408
minMSD: 0.609, avgMSD: 1.186
rF: 2.087, DAO: 22.189, DAC: 0.934
.
.
.
Working on test 10/10, batch 584/584... Complete.
minADE3: 0.606, minFDE3: 1.151
avgADE3: 1.185, avgFDE3: 2.407
minMSD: 0.606, avgMSD: 1.185
rF: 2.091, DAO: 22.162, DAC: 0.934
--Final Performane Report--
minADE3: 0.608±0.00136, minFDE3: 1.152±0.00204
avgADE3: 1.184±0.00091, avgFDE3: 2.405±0.00232
minMSD: 0.608±0.00136, avgMSD: 1.184±0.00091
rF: 2.087±0.00311, DAO: 22.168±0.01221, DAC: 0.934±0.00019
```

## Things to do

- [x] Re-implement preprocess_argoverse.py and preprocess_nuscenes.py.
- [ ] Re-organize implementations for MATF/R2P2/Desire/CSP/SimpleEncoderDecoder
- [ ] Re-organize comments for model definition and miscellaneous scripts.
- [ ] Note hyperparameters for training/fine-tuning at README.md.

## Citation
Please cite the original publication;
```
@inproceedings{park2020diverse,
  title={Diverse and admissible trajectory forecasting through multimodal context understanding},
  author={Park, Seong Hyeon and Lee, Gyubok and Seo, Jimin and Bhat, Manoj and Kang, Minseok and Francis, Jonathan and Jadhav, Ashwin and Liang, Paul Pu and Morency, Louis-Philippe},
  booktitle={European Conference on Computer Vision},
  pages={282--298},
  year={2020},
  organization={Springer}
}
```


## License

This code is published under the [General Public License version 2.0](LICENSE).
