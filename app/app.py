from flask import Flask, request, render_template, jsonify
from model import run_model, model_fetch

app = Flask(__name__)
app.static_folder = 'static'


model = run_model()


def chatbot_response(x):
   return model_fetch(model,x)
   #return x

@app.route('/')
def index():
   greeting = 'Ahoy there matey'
   return render_template('yergu1.html', greeting = greeting)

@app.route('/predict', methods=['POST'])
def make_prediction():
   # get JSON payload from client
   data = request.get_json()
   print("you posted {0}".format(data['entry']))

   # send a JSON response to the client
   return jsonify(response=chatbot_response(data['entry']))

if __name__ == '__main__':
   app.run(host='0.0.0.0', port = 5000)