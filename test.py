from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import numpy as np
from backend import backend_model
app = Flask(__name__)



# run this once
# --------
m=backend_model.Model()
m.prepare_model()
# --------

@app.route("/", methods=["POST","GET"])
def home():
    if request.method =="POST":
        text = request.form["txt"]
        return redirect(url_for("result", inp=text))
    else:
        return render_template("web.html")

@app.route("/<inp>", methods=["POST","GET"])
def result(inp):
    page_no = request.args.get('page', 0, type=int)
    page_size = request.args.get('pageSize', 5, type=int)
    articles, articles_urls, score, region=m.retrieve_documents(inp,10,"poynter")
    src100=np.array(score)*100
    #combine all the data we need into a set
    articles_datas=np.vstack((articles,articles_urls,np.round(src100,2),region)).T
    found=True
    if articles==[]:
        found=False
    pagination = Pagination(page_no, articles_datas, page_size=page_size)
    if request.method =="POST":
        text = request.form["txt"]
        #if the input is empty, stay on the same page
        if text!="":
            return redirect(url_for("result", inp=text))
        else:
            return render_template("result.html", pagination=pagination,input=inp, found=found)
    else:
        return render_template("result.html",pagination=pagination,input=inp, found=found)




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
        _pages = (self.total + page_size - 1) / page_size
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
