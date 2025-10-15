import pandas as pd

def load_dataset(path):
    return pd.read_csv(path)

def save_predictions(y_true, y_pred, output_path="../../lr_predictions.csv"):
    pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
