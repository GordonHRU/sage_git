import numpy as np
import pandas as pd
import tensorflow as tf
# assert tf.__version__.startswith('2')

# plot_model
# pip install pydot
# pip install pydotplus
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

# pip install scikit-learn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import os, random, shutil, warnings, cv2  # pip install opencv-python
import json
import pickle
import joblib
from azureml.core.model import Model
from azureml.core import Workspace, Run
import logging




def init():
    logging.info("starting script")
    global model
    # retreive the path to the model file using the model name

    # workspacename = Workspace(subscription_id = "Azure_account_1", resource_group = "Machine_Learning", workspace_name = "chinese_herbs_ML")
    # model_list = Model.list(workspace = workspacename)
    # logging.info("the model list"+model_list)


    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
    )
    # model_path = Model.get_model_path(model_name = 'model_test_1', version=1, _workspace="701dde0c-162d-4763-8137-1bc194247bb3")
    model = joblib.load(open(model_path, 'rb'))

    # model_name = 'model_test_1'
    # ws = Run.get_context().experiment.workspace
    # model_obj = Model(ws, model_name )
    # model_path = model_obj.download(exist_ok = True)
    # model = joblib.load(model_path)
def run(raw_data):
    # data = np.array(json.loads(raw_data)['data'])
    # make prediction
    logging.info("starting script")
    y_hat = model.evaluate(raw_data)
    return json.dumps(y_hat.tolist())


from flask import Flask
from flask.views import MethodView
from PIL import Image
import io

app = Flask(__name__)


class API_Test(MethodView):
    # def get(self):
    #     return jsonify(message='I am GET')

    def post(self, raw_data):
        image = Image.open(io.BytesIO(raw_data))
        model_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
        )
        model = joblib.load(open(model_path, 'rb'))
        y_hat = model.evaluate(image)
        return y_hat
        
    # def delete(self):
    #     return jsonify(message='I am DELETE')


app.add_url_rule('/test_api/', view_func=API_Test.as_view('test_api'))

if __name__ == '__main__':
    app.run(debug=True)