from flask import Flask
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

@app.route("/")
def controller():
    model_path = ""
    predict_model = ""
    
    

if __name__ == "__main__":
    app.run()