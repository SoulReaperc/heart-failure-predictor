from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form.get(x)) for x in [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]]
        final_input = np.array([features])
        prediction = model.predict(final_input)[0]
        output = "High Risk" if prediction == 1 else "Low Risk"
        return render_template("index.html", prediction_text=f"Heart Failure Risk: {output}")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
