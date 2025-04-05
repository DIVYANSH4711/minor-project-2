from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


@app.route('/api/allSongs', methods=['GET'])
def get_data():
    return jsonify({'data': [1, 2, 3, 4]})

if __name__ == '__main__':
    app.run(debug=True)
