from flask import Flask, redirect, url_for, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def home():
    articles=data[:10]
    if request.method =="POST":
        text = request.form["txt"]
        return redirect(url_for("result", inp=text))
    else:
        return render_template("web.html", twitter_data=articles)



@app.route("/<inp>", methods=["POST","GET"])
def result(inp):
    out,articles=find(inp)
    if request.method =="POST":
        text = request.form["txt"]
        #if the input is empty, stay on the same page
        if text!="":
            return redirect(url_for("result", inp=text))
        else:
            return render_template("result.html", out=out, data=articles)
    else:
        return render_template("result.html", out=out, data=articles)



#test model
#reading first five article from csv file.
poynter_data_path="backend/data/poynter_claims_explanation.csv"
poynter_df=pd.read_csv(poynter_data_path).dropna() # Remove missing values.
# poynter_df=pd.read_csv(poynter_data_path).iloc[:,1]
poynter_df=pd.read_csv(poynter_data_path).iloc[:20,1]
data=list(set(poynter_df.values))


# if the input text contains the terms of "covid 19", we say it is true
def find(text):
    result="Fake"
    relative_art=["hello"]
    words = text.split(" ")
    for i in range(len(words)-1):
        if words[i].lower() =="covid":
            if words[i+1].lower() =="19":
                result="Fact"
                
    # assume these articles are the relative articles
    relative_art=data[:6]     
    return result,relative_art
 

if __name__ == "__main__":
    app.run(debug=True)
