import itertools
import pickle
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from rdt.transformers import OneHotEncoder, GaussianNormalizer, FloatFormatter
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
from rdt import HyperTransformer
from diffusion import DDPM

# paths
<<<<<<< HEAD
BASE = Path("...")
=======
BASE = Path(__file__).resolve().parents[2]
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
DATA_PATH = BASE / "data" / "EV_Charging_Data_processed.csv"
MODEL_DIR = BASE / "diffusion" / "models"
PLOT_DIR  = BASE / "diffusion" / "plots"
for _d in (MODEL_DIR, PLOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# columns
COND_COLS = ["season", "weekday_group", "location_group"]
OUTPUT_FOCUS = ["plugin_hour", "connection_time", "energy_session"]
OTHER_CATS = ["plugin_day", "plugin_month"]
OTHER_CONT = [ "electricity_price","temperature","humidity","solar_radiation","wind_speed"]
ALL_COLS = COND_COLS + OUTPUT_FOCUS + OTHER_CATS + OTHER_CONT

# data split
def stratified_split(df):
    df = df.copy()[ALL_COLS]
    df["strata"] = (df["location_group"].astype(str)+ "_"+ df["season"].astype(str)+ "_"+ df["weekday_group"])
    tr_val, _ = train_test_split(df, test_size=0.15, random_state=306, stratify=df["strata"])
    train_set, _ = train_test_split( tr_val, test_size=0.1765, random_state=306, stratify=tr_val["strata"])
    train_set.drop(columns=["strata"], inplace=True)
    return train_set.reset_index(drop=True)

# preprocessing with hypertransformer
def fit_preprocessors(train_df):
    ht = HyperTransformer()
    
    sdtypes = {
        'season': 'categorical',
        'weekday_group': 'categorical',
        'location_group': 'categorical',
        'plugin_hour': 'numerical', 
        'plugin_day': 'categorical',
        'plugin_month': 'categorical',
        'connection_time': 'numerical',
        'energy_session': 'numerical',
        'electricity_price': 'numerical',
        'temperature': 'numerical',
        'humidity': 'numerical',
        'solar_radiation': 'numerical',
        'wind_speed': 'numerical'}
    
    transformers = {
        'season': OneHotEncoder(),
        'weekday_group': OneHotEncoder(),
        'location_group': OneHotEncoder(),
        'plugin_hour': GaussianNormalizer(),  
        'plugin_day': OneHotEncoder(),
        'plugin_month': OneHotEncoder(),
        'connection_time': GaussianNormalizer(),
        'energy_session': GaussianNormalizer(),
        'electricity_price': FloatFormatter(),
        'temperature': FloatFormatter(),
        'humidity': FloatFormatter(),
        'solar_radiation': FloatFormatter(),
        'wind_speed': FloatFormatter()}
    
    ht.set_config({
        'sdtypes': sdtypes,
        'transformers': transformers
    })
    
    ht.fit(train_df)
    transformed = ht.transform(train_df)
    
    condition_trans_cols = []
    for col in COND_COLS:
  
        cols = [c for c in transformed.columns if c.startswith(f"{col}.")]
        if cols:
            condition_trans_cols.extend(cols)
        else:
            condition_trans_cols.append(col)
    
    with open(MODEL_DIR / 'hyper_transformer.pkl', 'wb') as f:
        pickle.dump((ht, condition_trans_cols), f)  

def transform(df):
    with open(MODEL_DIR / 'hyper_transformer.pkl', 'rb') as f:
        ht, condition_trans_cols = pickle.load(f)  #  
    
    transformed = ht.transform(df)
    valid_condition_cols = [col for col in condition_trans_cols if col in transformed.columns]
    cond_data = transformed[valid_condition_cols].values.astype(np.float32)
    input_data = transformed.drop(columns=valid_condition_cols).values.astype(np.float32)
    return input_data, cond_data

class FullDataset(Dataset):
    def __init__(self, input_data, cond_data):
        self.input_data = torch.from_numpy(input_data)  
        self.cond_data = torch.from_numpy(cond_data)   

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return {
            "input": self.input_data[idx].unsqueeze(0),  
            "condition": self.cond_data[idx]            
        }



def main():
    df = pd.read_csv(DATA_PATH, usecols=ALL_COLS)
    train_df = stratified_split(df)
    fit_preprocessors(train_df)
    X_train, C_train = transform(train_df)

    input_dim = X_train.shape[1]
    cond_dim = C_train.shape[1]

    train_dataset = FullDataset(X_train, C_train)
    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # over different depths  and attention heads
    depths = [1,2,3,4]  
    heads  = [None, 2, 4, 8] 
    combinations = list(itertools.product(depths, heads))
    total = len(combinations)
        
    for i, (depth, nhead) in enumerate(combinations):
        print(f"\nTraining model {i+1}/{total}: depth={depth}, heads={nhead}")
        
        config_dict = {
            "network": "attention" if nhead is not None else "mlp",
            "input_dim": input_dim,
            "cond_dim": cond_dim,
            "hidden_dim": 128,  
            "nhead": 0 if nhead is None else nhead,
            "seq_len": 1,
            "device": DEVICE,
            "n_steps": 1000,  
            "schedule": "linear", 
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "init_lr": 1e-3,
            "n_epochs": 100, 
            "depth": depth
        }
        
        opt = SimpleNamespace(**config_dict)
        
        config_dict["device"] = str(config_dict["device"])
        tag = f"d{depth}_h{nhead if nhead is not None else 'none'}"
        with open(MODEL_DIR / f"config_{tag}.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        model = DDPM(opt, loader)
        losses = model.train()

        with open(MODEL_DIR / f"model_{tag}.pkl", "wb") as f:
            pickle.dump(model, f)

        # plot losses
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve – depth={depth}, heads={nhead if nhead is not None else 'none'}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"loss_curve_{tag}.png")
        plt.close()
        
        print(f"Completed model {i+1}/{total}")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> ee9996e (added new codes/data used in adding new analysis)
