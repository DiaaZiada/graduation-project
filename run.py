from flask import Flask, escape, request, jsonify
from utils import train, predict, video_to_emb
import json

app = Flask(__name__)




@app.route('/new_student', methods=['POST'])
def new_student():
    video_path = request.json["video_path"]
    id = request.json["id"]
    video_to_emb(video_path,id)
    return 'done'


@app.route('/make_section', methods=['POST'])
def make_section():
    ids = request.json["ids"]
    group_name = request.json["group_name"]
    print(ids)
    print(group_name)
    train(ids, group_name)
    return "done"

@app.route('/predict', methods=['POST'])
def url_predict():
    image = request.json["image"]
    group_name = request.json["group_name"]
    print(image, group_name)
    out = predict(image, group_name)
    data = {
        "label":str(out[1][0]),
        "acc":str(out[0])
    }
    print(data)
    return jsonify(data)
if __name__ == "__main__":
    app.run(debug=True)
