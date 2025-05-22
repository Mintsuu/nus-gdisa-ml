import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('dt_model.pkl', 'rb'))

@app.route('/predict',methods=['GET'])
def predict():
    '''
    For direct API calls trought request
    '''
    #data = request.get_json(force=True)
    name = request.args.get('name', 'Guest')  # Default to 'Guest' if 'name' is not provided
    age = request.args.get('age', 0)  # Default to 'unknown' if 'age' is not provided
    hasJob = request.args.get('hasJob')
    ownHouse = request.args.get('ownHouse')
    creditRating = request.args.get('credit',1)

    prediction = model.predict([[age,hasJob,ownHouse,creditRating]])
    output = prediction[0]

    # Create a response dictionary
    response = {
        'message': f'Hello, {name}! your loan application is likely to be {output}'
    }
    return jsonify(response)


'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    # For direct API calls trought request
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

'''
if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)



