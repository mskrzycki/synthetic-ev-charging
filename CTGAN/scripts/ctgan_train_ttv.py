import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from custom_ctgan import CustomCTGAN
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt
import os

BASE = Path("...")
DATA   = BASE / "data" / "EV_Charging_Data_processed.csv"  
SETS   = BASE / "CTGAN" / "sets"  
MOD    = BASE / "CTGAN" / "models"  
LOSS_PLOTS = MOD / "loss_plots"

SETS.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
LOSS_PLOTS.mkdir(parents=True, exist_ok=True)

EPOCHS = 300
BATCH  = 500
DEPTHS = [1,2,3,4]       
HEADS  = [None, 2, 4, 8]    

COLS = [
    "location_group", "weekday_group", "plugin_hour", "plugin_day",
    "plugin_month", "area_type", "electricity_price",
    "temperature", "humidity", "solar_radiation", "wind_speed",
    "season", "connection_time", "energy_session"]

df = pd.read_csv(DATA)[COLS].copy()

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

for col in [
    'plugin_hour', 'plugin_day', 'plugin_month',
    'season', 'location_group', 'weekday_group'
]:
    metadata.update_column(col, sdtype='categorical')

for col in [
    'connection_time','energy_session','electricity_price',
    'temperature','humidity','solar_radiation','wind_speed'
]:
    metadata.update_column(col)


def train():
    df["strata"] = (df["location_group"].astype(str)+ "_" + df["season"].astype(str)+ "_" + df["weekday_group"])
    tr_val, test_set = train_test_split(df, test_size=0.15, random_state=306, stratify=df["strata"])
    train_set, val_set = train_test_split(tr_val, test_size=0.1765, random_state=306, stratify=tr_val["strata"])

    for d in DEPTHS:
        for h in HEADS:
            tag = f"d{d}_{'off' if h is None else str(h)+'h'}"
            model_path = MOD / f"model_{tag}.pkl"
            if model_path.exists(): # if already trained, don't train again
                continue
            
            ctgan = CustomCTGAN(
                generator_dim=[128] * d, # depth
                discriminator_dim=[128] * d, # depth
                cross_att_heads=h, # num of heads
                epochs=EPOCHS,
                batch_size=BATCH,
                verbose=True,
                metadata=metadata
            )

            print(f"Train: depth={d}, heads={h}")
            ctgan.fit(train_set.drop(columns="strata"))
        
            gen = ctgan.get_loss_values()['Generator Loss']
            disc = ctgan.get_loss_values()['Discriminator Loss']

            # save losses
            if gen is not None and disc is not None:
                df_losses = pd.DataFrame({
                    'epoch': range(1, len(gen) + 1),
                    'generator_loss': gen,
                    'discriminator_loss': disc})
                out_csv = LOSS_PLOTS / f"{tag}_losses.csv"
                df_losses.to_csv(out_csv, index=False)

    # save sets
    train_set.to_csv(SETS / "train.csv", index=False)
    val_set.to_csv(SETS / "val.csv", index=False)
    test_set.to_csv(SETS / "test.csv", index=False)

if __name__ == "__main__":
    train()