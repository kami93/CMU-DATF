import os

TRAIN = True
def update_config(configs):
    if TRAIN:
        return update_config_train(configs)
    else:
        return update_config_test(configs)

def update_config_test(configs):
    # Dataset
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = (64, 64)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 
    configs.device = "cuda"

    # Model
    configs.model_name = "Global_Scene_CAM_NFDecoder"
    configs.TAG = "global_scene_nfdecode"
    configs.agent_embed_dim = 128
    configs.nfuture = 12 # redundant if using sampling ratesa
    configs.att_dropout = 0.1
    configs.lstm_layers = 1
    configs.lstm_dropout = 0.3
    configs.velocity_const = 0.5
    configs.num_candidates = 12
    configs.decoding_steps = 6
    configs.att = True
    configs.ploss_type = "map"
    configs.ploss_criterion = "Interpolated_Ploss"

    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"
    configs.test_ckpt = "../checkpoints/carla_MATF_D128.pth.tar"

    # Test
    configs.batch_size = 1
    configs.test_set = False 
    return configs

def update_config_train(configs):

    # Dataset
    configs.device = "cuda"
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = (64, 64)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 

    # Model
    configs.model_name = "Global_Scene_CAM_NFDecoder"
    configs.TAG = "global_scene_nfdecode"
    configs.agent_embed_dim = 128
    configs.nfuture = 12 # redundant if using sampling ratesa
    configs.att_dropout = 0.1
    configs.lstm_layers = 1
    configs.lstm_dropout = 0.3
    configs.velocity_const = 0.5
    configs.num_candidates = 12
    configs.decoding_steps = 6
    configs.att = True
    configs.ploss_type = "map"
    configs.ploss_criterion = "Interpolated_Ploss"

    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    configs.optimizer_name = "adam"
    configs.batch_size = 4
    configs.train = True
    configs.validate = True
    return configs

if __name__=="__main__":
    print("Call update_config method")
