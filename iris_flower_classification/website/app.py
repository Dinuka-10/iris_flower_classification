from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = 'model/iris.pickle'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    model = None

flower_map = {
    "setosa": ("Setosa", "setosa.jpg"),
    "versicolor": ("Versicolor", "versicolor.jpg"),
    "virginica": ("Virginica", "virginica.jpg")
}

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_name = None
    pred_img = None

    if request.method == 'POST':
        try:
            features = [
                float(request.form['sepallength']),
                float(request.form['sepalwidth']),
                float(request.form['petallength']),
                float(request.form['petalwidth'])
            ]

            features_arr = np.array(features).reshape(1, -1)
            
            prediction = model.predict(features_arr)[0]

            pred_name, pred_img = flower_map.get(prediction.lower(), ("Unknown", None))

        except Exception as e:
            print(f"Error during prediction: {e}")
            pred_name = "Invalid Input"
            pred_img = None

    return render_template(
        'index.html',
        pred_name=pred_name,
        pred_img=pred_img
    )

if __name__ == '__main__':
    app.run(debug=True)