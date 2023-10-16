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
            "drug_disease_pd_dir": "test.csv",
            "max_path_length": 8, 
            "max_path_num": 256,
            "random_state": 0
        }
    },

    "K": 5,

    "trainer": {
        "save_dir": "saved/",
        "verbosity": 2
    }}
    train_json['arch']['args']['deepm'] = model
    train_json['loss'] = loss
    with open("config/" + config_name + "_inference.json", "w") as f:
        json.dumps(train_json, f, ensure_ascii=False)
    return "config/" + config_name + ".json"

if __name__ == "__main__":
    filename = generate_config()
    print("your config file is saved at " + filename)