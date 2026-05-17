from src.models.registry import get_models
from src.models.train import train_model
from src.models.evaluate import evaluate

def run_all_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training: {name}")
        
        model = train_model(model, X_train, y_train)
        score = evaluate(model, X_test, y_test)
        
        results[name] = score

    return results