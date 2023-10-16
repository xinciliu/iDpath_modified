import sys
import json

def generate_config():
    model = sys.argv[1]
    loss = sys.argv[2]
    config_name = sys.argv[3]
    train_json = {
    "name": "iDPath_Modified",
    "n_gpu": 1,
    "arch": {
        "type": "iDPath_Modified",
        "args": {
            "emb_dim": 64,
            "gcn_layersize": [64, 128, 64],
            "dropout": 0.5,
            "deepm": "lstm"}
    },
    "data_loader": {
        "type": "PathDataLoader",
        "args":{
            "data_dir": "data/",
            "batch_size": 512,
            "max_path_length": 8, 
            "max_path_num": 256, 
            "random_state": 0,
            "recreate": false,
            "use_disease_seed": true,
            "shuffle": true,
            "validation_split": 0.1,
            "test_split": 0.2,
            "num_workers": 2
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "bce_withlogits_loss",
    "metrics": [
        "accuracy", "recall", "roc_auc", "pr_auc", "f1_score"
    ],
    "Ks": [5, 10, 20, 50, 100],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 50,

        "save_dir": "saved/",
        "save_period": 50,
        "verbosity": 2,
        
        "monitor": "min val_loss",
        "early_stop": 10,

        "tensorboard": false
    }}
    train_json['arch']['args']['deepm'] = model
    train_json['loss'] = loss
    with open("config/" + config_name + ".json", "w") as f:
        json.dumps(train_json, f, ensure_ascii=False)
    return "config/" + config_name + ".json"

if __name__ == "__main__":
    filename = generate_config()
    print("your config file is saved at " + filename)