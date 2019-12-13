from flask import Flask, render_template, request, url_for
from api import text_generator
import re
import os

# Initialize flask app
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def generate():
    if request.method == 'POST':
        input_text = request.form['message']
        geneterated_text = text_generator(input_text)
    else:
        input_text = " "
        geneterated_text = " "

    return render_template('index.html',
                           seed = input_text,
                           generation = geneterated_text)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
