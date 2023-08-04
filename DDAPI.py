import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from pyngrok import ngrok

app = Flask(__name__)

port = 8080
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
app.config["BASE_URL"] = public_url

raw_data = pd.read_csv(r'/home/mateus/Desktop/Estudo/VSCode Projects/DiabetesDetection/diabetes-detection/diabetes-dataset.csv')

train_data = raw_data.drop(columns = ['Insulin', 'DiabetesPedigreeFunction'])

X = train_data.drop(columns = 'Outcome')
Y = train_data['Outcome'].astype(str)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

user_clf = RandomForestClassifier(max_depth=20, max_features=4, random_state=42)
user_clf = user_clf.fit(X_train,Y_train)

@app.route("/detect_diabetes", methods=[ 'GET'])

def detect_diabetes():
    pregnancies, glucose, blood_pressure, skin_thickness, height, weight, age = request.args.get("pregnancies"), request.args.get("glucose"), request.args.get("blood_pressure"), request.args.get("skin_thickness"), request.args.get("height"), request.args.get("weight"), request.args.get("age")
    data = [
        int(pregnancies),
        int(glucose),
        int(blood_pressure),
        int(skin_thickness),
        float(weight)/((float(height)/100)**2),
        int(age)
    ]

    print(data)

    data = pd.DataFrame([data], columns = ['Pregnancies', 'Glucose',	'BloodPressure',	'SkinThickness',	'BMI',	'Age'])
    
    probability = user_clf.predict_proba(data)[0][1]

    result = f"De acordo com nossos parâmetros, há uma chance de {probability*100:.2f}% de que você tenha diabetes. "
    if probability > 0.5:
        result = (result + "Você deve procurar um médico para realizar exames mais detalhados.")
    print(result)

    return jsonify({"result":result}), 200


if __name__ == '__main__':
    app.run(port=8080)
