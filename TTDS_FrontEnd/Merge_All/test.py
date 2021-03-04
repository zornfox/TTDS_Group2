from flask import Flask, redirect, url_for, render_template, request
import pandas as pd

app = Flask(__name__)

#test model
#reading first five article from csv file.
#poynter_data_path="backend/data/poynter_claims_explanation.csv"
poynter_data_path="backend/data/poynter_claims_explanation.csv"
poynter_df=pd.read_csv(poynter_data_path).dropna() # Remove missing values.
# poynter_df=pd.read_csv(poynter_data_path).iloc[:,1]
poynter_df=pd.read_csv(poynter_data_path).iloc[:20,1]
datas=list(set(poynter_df.values))


@app.route("/", methods=["POST","GET"])
def home():
    if request.method =="POST":
        text = request.form["txt"]
        return redirect(url_for("result", inp=text))
    else:
        return render_template("web.html", twitter_data=datas[:10])

@app.route("/<inp>", methods=["POST","GET"])
def result(inp):
    page_no = request.args.get('page', 0, type=int)
    page_size = request.args.get('pageSize', 5, type=int)
    out,articles=find(inp)
    pagination = Pagination(page_no, articles, page_size=page_size)
    if request.method =="POST":
        text = request.form["txt"]
        #if the input is empty, stay on the same page
        if text!="":
            return redirect(url_for("result", inp=text))
        else:
            return render_template("result.html", pagination=pagination,out=out)
    else:
        return render_template("result.html",pagination=pagination,out=out)


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
    relative_art=datas
    return result,relative_art



class Pagination(object):
    """
    paginate.page current page
    paginate.pages total page
    paginate.total total data
    """

    def __init__(self, page,datas=[],page_size=10):
        try:
            current_page = int(page)
        except Exception as e:
            current_page = 1
        if current_page <= 0:
            current_page = 1

        self
        ## 10 per page
        self.page_size = page_size
        # current page 
        self.page = current_page
        # total data size 
        self.total = len(datas)
        # total page 
        _pages = (self.total + page_size - 1) / page_size;
        self.pages = int(_pages)
        self.has_prev = current_page > 1 and  current_page <= self.pages if True else False
        self.has_next = current_page < self.pages if True else False
        start_index = (self.page-1)*self.page_size
        end_index = self.page*self.page_size
        self.items = datas[start_index:end_index]

    @property
    def prev_num(self):
        if self.has_prev:
            return int(self.page - 1)
        else:
            return self.pages

    @property
    def next_num(self):
        if self.has_next:
            return int(self.page + 1)
        else:
            return self.pages



    def iter_pages(self):
        for num in range(1, self.pages + 1):
                yield num

if __name__ == "__main__":
    app.run(debug=True)
