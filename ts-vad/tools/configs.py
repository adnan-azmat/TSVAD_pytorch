configs_4Speakers_wavlm = {"input_dim": 40,
"average_pooling": 3,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"fea_dim": 768,
"n_heads1": 8,
"embedding_path1": "/home/users/ntu/adnan002/scratch/repos/Adnan/2/TSVAD_pytorch/ts-vad/pretrained_models/cluster_center_128.npy",
"n_heads2": 8,
"embedding_path2": "/home/users/ntu/adnan002/scratch/repos/Adnan/2/TSVAD_pytorch/ts-vad/pretrained_models/xvector_cluster_center_128.npy",
"splice_size": 768+100+192+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_4Speakers_evector256_wavlm = {"input_dim": 40,
"average_pooling": 3,
"fea_dim": 768,
"n_heads1": 8,
"embedding_path": "/home/users/nus/e1342275/scratch/new_sim/data/ecapa_mem_mod_256.npy",
"n_heads2": 8,
"splice_size": 768+192+192,
"Linear_dim": 384,
"output_speaker": 4
}

configs = {
    "configs_4Speakers_wavlm": configs_4Speakers_wavlm,
    "configs_4Speakers_evector256_wavlm": configs_4Speakers_evector256_wavlm
    }