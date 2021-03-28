import os
import re
import pandas as pd
from stemming.porter2 import stem
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import csv
import pandas as pd

# Path to stop words file
stop_word_path="englishST.txt"

# Load in data
data=[]
poynter_data_path="data/poynter_claims_explanation.csv"
poynter_df=pd.read_csv(poynter_data_path).dropna()
poynter_df=pd.read_csv(poynter_data_path).iloc[:,1]

data=list(set(poynter_df.values))

# w2v path
covid_w2v_path = "models/model.bin"
all_w2v_path = "models/all_model.bin"
all_train_w2v_path = "models/all_model_train.bin"

with open(stop_word_path, "r") as f:
    stop_words=f.read()

print(covid_w2v_path)

covid_model = Word2Vec.load(covid_w2v_path)
all_model = Word2Vec.load(all_w2v_path)
all_model_train= Word2Vec.load(all_train_w2v_path)

def tokenize(string_input):
    # split on any non-letter character
    string_input=string_input.replace("’","").replace("‘","").replace(".","").replace("-","").replace("'","").replace(",","").replace("\"", "")
    split_strings=r"[^a-zA-Z0-9$£#]+"
    tokened=re.split(split_strings,string_input)
    return tokened

def case_fix(string_input):
    # lowercase all words
    return [w.lower() for w in string_input]

def remove_stop(string_input):
    # include only words which are not in stop_words
    return [w for w in string_input if w not in stop_words]

def porter_stem(string_input):
    # stem words
    return [stem(w) for w in string_input]

def preprocess(strings):
    # pipeline of preprocessing
    tokens=tokenize(strings)
    lowers=case_fix(tokens)
    stopped=remove_stop(lowers)
    return stopped


# Preprocess data
preprocessed_data=[]
for t in data:
    preprocessed_data.append(preprocess(t))

data=dict(list(zip(range(len(preprocessed_data)),preprocessed_data)))
# dict of list of string<vclaiim>
# e.g. {997: ['website','offers','rapid','diagnose'], 998:['sds','12ewe']...}

# --------------- evaluation codes start ------------------------
# test data directory: data/CheckThat2020
# If not in evaluation case, just comment out this block

# parse the verified claims
vclaims_directory = 'data/TestData/verified_claims.docs.tsv'
vclaims_fields = ['vclaim_id', 'vclaim', 'title']
df_vc = pd.read_csv(vclaims_directory, usecols = vclaims_fields, sep = '\t')
vclaims_list = df_vc.vclaim.tolist().copy()
vclaims_tokens=[]
for t in vclaims_list:
    vclaims_tokens.append(preprocess(t))

vclaims_id = df_vc.vclaim_id.tolist().copy()
data = dict(zip(vclaims_id, vclaims_tokens)) # replace data variable for evaluation

# prepare queries and put them in a list
tweets_directory = 'data/TestData/train/tweets.queries.tsv'
tweets_fields = ['tweet_id', 'tweet_content']
df_t = pd.read_csv(tweets_directory, usecols = tweets_fields, sep = '\t')
tweets_list = df_t.tweet_content.tolist().copy()
tweets_id = df_t.tweet_id.tolist().copy()

# ------------------- evaluation codes end ---------------------



class II():
    def __init__(self,stop_words_path,data):
        self.data=data
        st=stop_words_path
        self.stop_words=None

        # read stop words from file
        with open(st, "r") as f:
            self.stop_words=f.read()
            self.stop_words=self.stop_words.split("\n")

        # initialise inverted index
        i=0
        self.inverted_index={}
        self.ids=[]
        cur_id=None
        self.doc_text={}
        self.wv=False

        # # iterate over all documents
        for (d_id, d) in self.data.items():
            cur_id=d_id
            text=d
            self.ids.append(cur_id)
            self.doc_text[cur_id]=text

            # update inverted index using headline and text for document "cur_id"
            self.inverted_index=self.get_inverted_index(text,cur_id)

    def tokenize(self,string_input):
        # split on any non-letter character
        string_input=string_input.replace("’","").replace("‘","").replace(".","").replace("-","").replace("'","").replace(",","")
        split_strings=r"[^a-zA-Z0-9$£#]+"
        tokened=re.split(split_strings,string_input)
        return tokened

    def case_fix(self,string_input):
        # lowercase all words
        return [w.lower() for w in string_input]

    def remove_stop(self,string_input):
        # include only words which are not in stop_words
        return [w for w in string_input if w not in self.stop_words]

    def preprocess(self,strings):
        # pipeline of preprocessing
        p=self.tokenize(strings)
        p=self.case_fix(p)
        p=self.remove_stop(p)
        return p

    def get_inverted_index(self,document,cur_id):
        document=np.array(document)
        # for the document, get all unique terms and their respective frequency in the document
        (unique, counts)=np.unique(document,return_counts=True)
        for i in (range(len(unique))):
            w=unique[i]
            c=counts[i]

            # locate positions of word in document
            positions=list(np.argwhere(document==w).flatten())

            # initialise key in inverted_index if doesn't exist
            if self.inverted_index.get(w)==None:
                self.inverted_index[w]={}
            self.inverted_index[w][cur_id]=[c,positions]
        return self.inverted_index
    
    def get_tf_vectors(self, t, d):
        if t not in self.w2v_model.wv.vocab:
            print("hit")
            return self.inverted_index[t][d][0]
        threshold=0.85
        count=0
        # for dw in self.doc_text:
        #     if dw=="nerf":
        #         print(self.doc_text)
        for dw in self.doc_text[d]:
            if dw not in self.w2v_model.wv.vocab:
                print("hit1")
                continue
            else:
                sim=self.w2v_model.similarity(t,dw)
                if sim>=threshold:
                    count+=1
                # if dist<threshold:
                #     print("hit")
                #     count=count+1
        if count==0:
            print("hit2")
        return count

    def TFIDF(self,t,d):
        N=len(self.ids)
        tft=np.log10(self.inverted_index[t][d][0])
        dft=len(self.inverted_index[t].keys())
        ldft=np.log10(N/dft)
        return (1+tft)*ldft
    
    def TFIDF_wv(self,t,d):
        N=len(self.ids)
        tft=np.log10(self.get_tf_vectors(t,d))
        dft=len(self.inverted_index[t].keys())
        ldft=np.log10(N/dft)
        return (1+tft)*ldft
    
    def w2v(self, terms):
        cos_scores={}
        t_vec = np.sum([new_model[t] for t in terms],axis=0)/len(terms)
        for d in self.ids:
            d_text=self.doc_text[d]
            d_vec=np.zeros(100)
            i=0
            for t in d_text:
                if t not in new_model.wv.vocab:
                    continue
                d_vec=d_vec+new_model[t]
                i=i+1
                
            d_vec = d_vec/i
            cos_sim = dot(t_vec, d_vec)/(norm(t_vec)*norm(d_vec))
            cos_scores[d]=cos_sim
        return cos_scores
                    
    def query_w2v(self,q):
        terms=self.tokenize(q)
        terms=self.remove_stop(terms)
        x=self.w2v(terms)
        return dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
        
    def get_docs_with_terms(self,terms):
        docs=[]
        for t in terms:
            if t not in self.inverted_index.keys():
                continue
            for d in list(self.inverted_index[t].keys()):
                if d not in docs:
                    docs.append(d)
        return docs

    def get_docs_with_terms1(self,terms):
        docs=[]
        for t in terms:
            for d in list(self.inverted_index[t].keys()):
                if d not in docs:
                    docs.append(d)
        return docs

    def parse_tfidf_query(self,q,wv=False,w2v_model=None):
        if wv:
            self.w2v_model=w2v_model
        weighted_docs={}
        self.wv=wv
        terms=self.preprocess(q)
        if wv:
            docs=self.ids
        else:
            docs=self.get_docs_with_terms(terms)
        for d in docs:
            cur_w=0
            for t in terms:
                if t not in self.inverted_index.keys():
                    continue
                # if term not in this docuemnt
                if self.inverted_index[t].get(d)==None:
                    continue
                else:
                    # compute the TFIDF for term t and document d
                    if (self.wv==True):
                        cur_w=cur_w+self.TFIDF_wv(t,d)
                    else:
                        cur_w=cur_w+self.TFIDF(t,d)
            weighted_docs[d]=cur_w
        sorted_docs = {k: v for k, v in sorted(weighted_docs.items(), key=lambda item: item[1], reverse = True)}
        return sorted_docs

    def get_II(self):
        return self.inverted_index

ii=II(stop_word_path,data)

# ================= normal codes start =========================

# # normal tfidf
# claim="the moon landing was"
# articles=list(ii.parse_tfidf_query(claim))
#
# # w2v tfidf with covid model
# claim="coronavirus is not a virus but a bacteria"
# articles=list(ii.parse_tfidf_query(claim,wv=True,w2v_model=covid_model))
#
# # w2v tfidf with full model
# claim="coronavirus is not a virus but a bacteria"
# articles=list(ii.parse_tfidf_query(claim,wv=True,w2v_model=all_model))
#
# print(articles[0:5])

# =================== normal codes end ==========================

# ----------------- evaluation codes start------------------------
# if not in evaluation case, please comment out this block and UNCOMMENT the block above this one

# prepare the result file
test_id = '04'
RankedIROutput = open(test_id + '_results.tsv', 'w', newline='')
results_fields = ['tweet_id','Q0','vclaim_id','rank','score','tag']
writer = csv.DictWriter(RankedIROutput, fieldnames = results_fields)
writer.writeheader()

for query_id, query in zip(tweets_id, tweets_list):

    search_result = list(ii.parse_tfidf_query(query,wv=True,w2v_model=all_model_train))

    # write into submitted file
    count = 0  # provide up to x result
    for matched_docID, toy_score in zip(search_result, list(reversed(range(len(search_result))))):
        # here list(reversed(range(len(search_result)))) are just fake scores generated for evaluation
        # can be replaced by real scores generated by model afterwards
        count = count + 1
        if count > 100:  # output top X result
            break

        tweet_id = query_id
        vclaim_id = matched_docID
        score = toy_score  # For simplicity we ignore the score for now
        tag = 'DC'  # meaningless tag
        return_data = {'tweet_id': tweet_id, 'Q0': 'Q0', 'vclaim_id': vclaim_id, 'rank': 1,
                       'score': score, 'tag': tag}
        writer = csv.DictWriter(RankedIROutput, fieldnames=results_fields, delimiter='\t')
        writer.writerow(return_data)

RankedIROutput.close()
# ----------------- evaluation codes end -------------------------
