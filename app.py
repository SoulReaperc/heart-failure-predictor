from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        if len(input_features) != 12:
            return render_template("index.html", prediction="❌ Please enter all 12 input values.", color_class="error")

        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            result = "Heart Failure Risk: High Risk"
            color_class = "high-risk"
        else:
            result = "Heart Failure Risk: Low Risk"
            color_class = "low-risk"

        return render_template("index.html", prediction=result, color_class=color_class)

    except Exception as e:
        return render_template("index.html", prediction=f"⚠️ Error: {str(e)}", color_class="error")


if __name__ == "__main__":
    app.run(debug=True)

