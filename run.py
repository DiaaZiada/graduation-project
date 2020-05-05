from flask import Flask, escape, request
from utils import train, predict, video_to_emb

app = Flask(__name__)




@app.route('/new_student/')
def new_student():
    video_path = request.args.get("video_path")
    id = request.args.get("id")
    video_to_emb(video_path,id)
    return "done"


@app.route('/make_section/')
def make_section():
    ids = request.args.get("ids")
    group_name = request.args.get("group_name")
    train(ids, group_name)
    return "done"

@app.route('/predict/')
def url_predict():
    image = request.args.get("image")
    group_name = request.args.get("group_name")
    out = predict(image, group_name)
    data = {
        "label":str(out[1][0]),
        "acc":str(out[0])
    }
    return data
if __name__ == "__main__":
    app.run(debug=True)