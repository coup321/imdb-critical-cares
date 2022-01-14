import pandas as pd
import numpy as np

def get_samples(model, dataset):
    X, y = list(dataset.as_numpy_iterator())
    preds = model.predict(X)
    
    df = pd.DataFrame([X, preds, y])

    return df