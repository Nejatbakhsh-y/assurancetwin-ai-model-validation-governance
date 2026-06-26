import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import joblib
import yaml
import requests
import tqdm
import fairlearn
import shap

print("Environment check passed.")
print(f"Python: {sys.version}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print("fairlearn imported successfully.")
print("shap imported successfully.")
