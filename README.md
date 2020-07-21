# Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding

This code is PyTorch implementation of our work, [Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding](https://arxiv.org/abs/2003.03212) (Seong Hyeon Park, Gyubok Lee, Manoj Bhat, Jimin Seo, Minseok Kang, Jonathan Francis, Ashwin R. Jadhav, Paul Pu Liang and Louis-Philippe Morency). 

![Model Diagram](model_figure.png)

Trajectory forecasting task was implemented using normalizing-flow, and achieving diverse while admissible trajectorys for vehicles.

## Dataset

You will have to download dataset into **data/[corresponding dataset]**, then verify the subdirectory of each dataset. Dataset should have structure same as:

```
-[dataset]
  |- 
```

Dataset will be extracted as cache at the initial execution. When not specified, cache file will be used for preceeding experiments. 

## Training

See details on *main.py*

To train proposed method;
```
python3 main.py \
--tag='Global_Scene_CAM_NFDecoder' --model_type='Global_Scene_CAM_NFDecoder' \
--dataset=nuscenes --batch_size=4 --num_epochs=100 --gpu_devices=3 \
--train_cache "./caches/nusc_train_cache.pkl" --val_cache ~/caches/nusc_val_cache.pkl \
--map_version '2.0' 
```


## Testing

Testing will be used by assigning checkpoint to argument *--test_ckpt*;
```
python3 main.py \
--tag "test-Global_Scene_CAM_NFDecoder" --model_type "Global_Scene_CAM_NFDecoder" \
--dataset argoverse --batch_size 64 --gpu_devices 0 \
--test_cache "./caches/argo_val_cache.pkl"  \
--test_partition 'val' --test_dir "./test/argoverse/AttGloScene-LocScene-CAM-NF" --test_ckpt "checkpoint.pth.tar"
```

## Things to do

- [x] ~~Select appropriate License; currently we used GPLv3.~~
- [ ] MATF_GAN had runtime error which has fixed. For coherence, this will be updated after recieving it.
- [ ] Check the *requirements*


## Citation
Please cite the original publication;

```
@article{park2020diverse,
  title={Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding},
  author={Park, Seong Hyeon and Lee, Gyubok and Bhat, Manoj and Seo, Jimin and Kang, Minseok and Francis, Jonathan and Jadhav, Ashwin R and Liang, Paul Pu and Morency, Louis-Philippe},
  journal={arXiv preprint arXiv:2003.03212},
  year={2020}
}
```


## License

This code is published under the [General Public License version 3.0](LICENSE).
