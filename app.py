# importing modules
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend on the base module FIRST
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)