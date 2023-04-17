import logging
from rasa.nlu.model import Interpreter
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the Rasa NLU model
model_path = './models/nlu-20230413-144513-tough-archway.tar.gz'
interpreter = Interpreter.load(model_path)

@app.route('/model/parse', methods=['POST'])
def predict():
    # Get the user input text from the request body
    text = request.json['text']
    
    # Use the NLU interpreter to parse the user input text
    result = interpreter.parse(text)
    
    # Return the NLU output as a JSON response
    return jsonify(result)

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True, port=8000)
