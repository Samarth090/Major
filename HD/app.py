from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Extracting data from form and converting it to a list of values
    float_features = [
        request.form['age'],
        request.form['sex'],
        request.form['chestPain'],
        request.form['bp'],
        request.form['cholesterol'],
        request.form['fbs'],
        request.form['ekg'],
        request.form['maxHr'],
        request.form['exerciseAngina'],
        request.form['stDepression'],
        request.form['slope'],
        request.form['vessels'],
        request.form['thallium']
    ]

    # Convert the inputs to a numpy array for prediction
    features = np.array([float_features], dtype=float)

    # Make the prediction
    prediction = model.predict(features)
    prediction_text = "The person has Heart Disease" if prediction[0] == 1 else "The person does not have Heart Disease"

    # Render the result on the same page
    return render_template("index.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
