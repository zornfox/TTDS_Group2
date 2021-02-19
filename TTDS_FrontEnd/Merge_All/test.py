from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def home():
    if request.method =="POST":
        text = request.form["txt"]
        return redirect(url_for("result", inp=text))
    else:
        return render_template("web.html")

@app.route("/<inp>", methods=["POST","GET"])
def result(inp):
    out=find(inp)
    if request.method =="POST":
        text = request.form["txt"]
        return redirect(url_for("result", inp=text))
    else:
        return render_template("result.html", out=out)



#test model
# if the input text contains the terms of "covid 19", we say it is true
def find(text):
    result="Fake"
    words = text.split(" ")
    for i in range(len(words)-1):
        if words[i].lower() =="covid":
            if words[i+1].lower() =="19":
                result="Fact"
                break
    return result


if __name__ == "__main__":
    app.run(debug=True)
