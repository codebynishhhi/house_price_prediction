import pandas as pd
import joblib

class InputAdapter:
    def __init__(self, defaults_path):
        self.defaults = joblib.load(defaults_path)

    def adapt(self, user_input: dict) -> pd.DataFrame:
        data = self.defaults.copy()

        # overwrite defaults with user input
        for k, v in user_input.items():
            data[k.replace("_", " ")] = v

        return pd.DataFrame([data])
