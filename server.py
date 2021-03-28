from flask import Flask, redirect, url_for, render_template, request
import numpy as np
from logger import logger
from model.model import Model
import time

app = Flask(__name__)

stop_word_path = 'data/englishST.txt'
poynter_data_path = 'data/poynter_title_url_region.csv'
cord19_data_path = 'data/cord19_titles.csv'
save_path = 'data/model.pickle'
covid_w2v_path = "data/models/model.bin"
all_w2v_path = "data/models/all_model.bin"

logger.info('Loading model ...')
start_time = time.time()
m = Model(stop_word_path, poynter_data_path, cord19_data_path, save_path,covid_w2v_path,all_w2v_path)
m.prepare_model()
logger.info("Load model use time: %.2f second" % (time.time() - start_time))
# --------


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        text = request.form["txt"]
        return redirect(url_for("result", inp=text, datatype="all"))
    else:
        return render_template("web.html")



@app.route("/<inp>/<datatype>", methods=["POST","GET"])
def result(inp,datatype):
    print("Search for: "+inp)
    page_no = request.args.get('page', 0, type=int)
    page_size = request.args.get('pageSize', 5, type=int)
    print('Using dataset',datatype)
    articles, articles_urls, score, region,titles=m.retrieve_documents(inp,100,datatype)
    src100=np.array(score)*100
    roundsrc=np.round(src100,2)
    found=True
    if articles==[]:
        found=False
        roundsrc=[]
    #combine all the data we need into a set
    articles_datas=np.vstack((articles,articles_urls,roundsrc,region,titles)).T

    pagination = Pagination(page_no, articles_datas, page_size=page_size)
    if request.method =="POST":
        currDataset  = request.form.get("dataset", type=str)
        text = request.form["txt"]
        #if the input is empty and try to change dataset, it will go to search for nothing
        if text=="":
            text=" "
        return redirect(url_for("result", inp=text, datatype=currDataset))
    else:
        return render_template("result.html",pagination=pagination,input=inp, found=found, dataset=datatype)


class Pagination(object):
    """
    paginate.page current page
    paginate.pages total page
    paginate.total total data
    """

    def __init__(self, page, datas=[], page_size=10):
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
        self.has_prev = current_page > 1 and current_page <= self.pages if True else False
        self.has_next = current_page < self.pages if True else False
        start_index = (self.page - 1) * self.page_size
        end_index = self.page * self.page_size
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
