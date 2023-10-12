from catboost import CatBoostClassifier
import pandas as pd

from flask import Flask, jsonify, request

# Load the model
model = CatBoostClassifier()      # parameters not required.
model.load_model('model.cbm')

# Init the app
app = Flask("default")


# Setup prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the provided JSON
    X = request.get_json()
    # Perform a prediction
    preds = model.predict(pd.DataFrame(X, index=[0]))[0, 1]
    # Output json with prediction
    result = {"predict": preds}
    return jsonify(result)


if __name__ == "__main__":
    # Run the app on local host and port 8989
    app.run(debug=True, host="0.0.0.0", port=8989)