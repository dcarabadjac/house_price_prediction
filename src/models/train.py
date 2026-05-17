from sklearn.model_selection import train_test_split
import pandas as pd

from src.models.registry import get_models
from src.models.evaluate import evaluate

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def run_training(config, model_name=None):
    # load processed data
    df = pd.read_csv(config["data"]["processed_path"])
    missing_target = df["price_per_sqm"].isna()
    if missing_target.any():
        print(f"Skipping {missing_target.sum()} rows with missing target price.")
        df = df[~missing_target]

    X = df.drop(columns=["price_per_sqm", "latitude", "longitude", "distance_to_sector_center"])
    y = df["price_per_sqm"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train"]["test_size"],
        random_state=42
    )
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")
 
    models = get_models()
    eval_errors = {}

    for name, model in models.items():
        if model_name and name != model_name:
            continue
        print(f"Training {name}...")
        
        trained_model = train_model(model, X_train, y_train)
        score = evaluate(trained_model, X_test, y_test)
        
        eval_errors[name] = score

    return eval_errors
