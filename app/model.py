import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

nltk.download('punkt')
 
import numpy
import tflearn
import tensorflow
import random


json_input = {
        "intents":[
        {
         "tag":"iTunes",
        "patterns": ["Music doesn't seem to work","Can't play any songs","Apple Music is stuck",'iTunes is slow'],
        "responses":["Try contacting our iTunes Store team here for more help: https://t.co/SDIe7UiyJN",
                    "Sorry to hear that. Please DM us with your apple ID and we will look into this"]
        },
        {
         "tag":"hardware",
        "patterns": ["button not working","home button","Slow after update"],
        "responses":["Oh, this seems like a hardware problem. Please reach out to us over DM",
                    "Uh-oh this doesn't seem like something we can solve over chat. Can you visit an apple service center"]
        },
        {
         "tag":"software",
        "patterns": ["app not working","camera bug","slow phone"],
        "responses":["Please check if restarting helps the issue?",
                    "Sorry to hear that. Let's start with a quick restart test?"]
        },
        {
        "tag": "opentoday",
         "patterns": ["Are you open today?", "When do you open today?", "What are your hours today?"],
         "responses": ["We're open every day from 9am-9pm", "Our hours are 9am-9pm every day"]
        },
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day","Hey"],
         "responses": ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"],
         "context_set": ""
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye"],
         "responses": ["See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon."]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
        {
         "tag": "AppleWatch",
         "patterns": ["My applewatch is not staying up long", "Battery life of Apple Watch is too less", "Apple Watch drains battery life"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"]
        },
        {
         "tag": "unknown",
         "patterns": [""],
         "responses": ["I'm afraid I do not have the answer to your query. Do not worry, this data is being monitored and used to teach me new stuff everyday!"]
        }
        ]
    }

words = []
labels = []
docs_x = []
docs_y = []

for intent in json_input['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

labels = sorted(labels)

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)



def run_model():

    # tensorflow.reset_default_graph()
    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    return model

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def model_fetch(model, text):
    results = model.predict([bag_of_words(text, words)])
    results_index = numpy.argmax(results)
    if results_index <0.5:
        tag = 'default'
    else:
        tag = labels[results_index]

    for tg in json_input["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

if __name__ == '__main__':
    print('Not to be executed seperately. Please make function call to run_model() model_fetch(model, text)')