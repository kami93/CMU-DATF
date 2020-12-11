import os

TRAIN = True
def update_config(configs):
    if TRAIN:
        return update_config_train(configs)
    else:
        return update_config_test(configs)

def update_config_train(configs):

    # Dataset
    configs.device = "cuda" 
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = [64, 64]
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 

    # Model
    configs.model_name = "DESIRE"
    configs.TAG = "desire"
    configs.ploss_type = "map"
    # configs.ploss = "Interpolated_Ploss"
    configs.num_candidates = 12
    configs.map_version = '2.0'
    nfuture = int(3 * configs.sampling_rate)
    configs.decoding_steps = nfuture
    configs.in_channels = 5 if configs.map_version == '2.1' else 3

    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    configs.optimizer_name = ["adam","adam"]
    configs.batch_size = 4
    configs.train = True
    configs.validate = True
    return configs


def update_config_test(configs):

    # Dataset
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = (60, 60)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 
    configs.device = "cuda"

    # Model
    configs.model_name = "DESIRE"
    configs.TAG = "desire"
    configs.ploss_type = "mseloss"
    configs.num_candidates = 12
    configs.map_version = '2.1'
    configs.decoding_steps = 6
    configs.in_channels = 5 if configs.map_version == '2.1' else 3
    
    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"
    configs.test_ckpt = "../checkpoints/carla_MATF_D128.pth.tar"

    # Test
    configs.batch_size = 1
    configs.test_set = False 
    return configs

if __name__=="__main__":
    print("Call update_config method")
