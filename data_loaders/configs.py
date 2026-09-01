dataset_config = {
    "cifar10": {
        "dataset_name": "cifar10",
        "root": "./data",
        "image_size": 224,
        "batch_size": 64,
        "val_split": 0.1
    },
    "mnist": {
        "dataset_name": "mnist",
        "root": "./data",
        "image_size": 224,
        "batch_size": 64,
        "val_split": 0.1
    },
    "CUB": {
        "dataset_name": "CUB",
        "batch_size": 64,
        "num_workers": 0,
        "weight_loss": True,  # Or False if we want to take into account class imbalance
        "sampling_percent": 1.0,  # Use 1.0 to load all concepts (no subsampling)
        "sampling_groups": False,  # True if we want group-based concept sampling
        "root_dir": '../data/CUB_200_2011/',
        "seed": 42,
        "output_dataset_vars": True,
        "image_size" : 224,
    },
    "CIFAR100_AwA2": {
        "dataset_name": "CIFAR100_AwA2",
        "batch_size": 64,
        "num_workers": 0,
        "seed": 42,
        "val_split": 0.1,
        "image_size": 32,
        "data_dir": '../data',
    },
    "AwA2": {
        "dataset_name": "AwA2",
        "batch_size": 64,
        "num_workers": 0,
        "seed": 42,
        "val_split": 0.1,
        "test_split": 0.2,
        "image_size": 224,
        "data_dir": '../data',
        "n_samples_per_class": None,
        "cv": False
    },
    "aPY": {
        "dataset_name": "aPY",
        "batch_size": 64,
        "num_workers": 0,
        "seed": 42,
        "val_split": 0.1,
        "test_split": 0.2,
        "image_size": 224,
        "data_dir": '../data',
        "n_samples_per_class": None,
        "cv": False,
        "to_crop":False,
    },
    "CIFAR100_aPY": {
        "dataset_name": "CIFAR100_aPY",
        "batch_size": 64,
        "num_workers": 0,
        "seed": 42,
        "val_split": 0.1,
        "image_size": 32,
        "data_dir": '../data',
    },
    
}


device_config = {"accelerator": "gpu",
                 "devices": "auto",
                 "num_sanity_val_steps": 0,
                 #"log_tool": "wandb",
}

training_config = {"opt": "Adam",
                    "lr": 1e-2,
                    "momentum": 0.9,
                    "weight_decay": 4e-5,
                    "max_epochs": 100,
                    "check_val_every_n_epoch": 5,
                    "lr_ratio": 2.,
                    "scheduler_type": "cosineannealing",  #LinearWarmupCosineAnnealingLR
                    "resume": False, 
                    "warm_epochs": 0,
                    "vib_beta": 0.00005,
                    "grad_ac_steps": 1,
                    "clip_grad_max_norm": 1.,
                    "disable_lr_scheduler": False,
                }


arch_config = {
    'CEM':{
        "backbone": "mobilenetv2",
        "c_loss_weight": 5,
        "y_loss_weight": 1,
        "emb_size": 16,
        "embedding_activation": "leakyrelu",
        "c2y_model": None,
        "c2y_layers": None,
        "shared_prob_gen": True,
        "concepts_weight_loss": None,
        "task_class_weights": None,
    },
    
    'prob_CBM':{
        "backbone": "mobilenetv2", #mobilenetv2 resnet18
        "model_type": "ProbCBM",
        "use_probabilsitic_concept": True,
        "pretrained": True,
        "pred_class": True,
        "pred_concept": True,
        "hidden_dim": 16,
        "class_hidden_dim": 128,
        "use_scale": True,
        "use_neg_concept": True,
        "train_class_mode": "joint",
        "activation_concept2class": "prob",
        "n_samples_inference": 50,
        "loss_weight_concept": 5.0,
        "loss_weight_class": 1.0,
        "vib_beta": 0.00005,
        "criterion_class": "ce",
        "criterion_concept": "MCBCELoss",#"bce",
    },
    
    'CT':{
        "attention_sparsity":0.5,  #sparsity penalty on attention
        "expl_lambda":1.0,  #weight of explanation loss
        "baseline": False,  #run baseline without concepts
        "ctc_model": 'cub_cvit',
        "n_unsup_concepts": 0,
        "n_spatial_concepts": 0,
        "num_heads": 12,
        "attention_dropout": 0.1,
        "projection_dropout": 0.1,
        "backbone": "vit_tiny_patch16_224",  #vit_large_patch16_224 
        }
        
}




intervention_config = {"training_intervention_prob": 0.25,
                       "active_intervention_values": None,
                       "inactive_intervention_values": None,
                       "intervention_policy": None,
                       "output_interventions": False,
                       "use_concept_groups": False,
    }