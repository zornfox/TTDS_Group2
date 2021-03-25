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

#url
#source
#region
#score
#

class Model():
    def __init__(self):
        self.stop_word_path="englishST.txt"
        self.poynter_data_path="data/poynter_title_url_region.csv"
        self.cord19_data_path="data/cord19_titles.csv"

        self.covid_w2v_path = "models/model.bin"
        self.all_w2v_path = "models/all_model.bin"

        # read stop words from file
        with open(self.stop_word_path, "r") as f:
            self.stop_words=f.read()
            self.stop_words=self.stop_words.split("\n")    
    
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

    def prepare_model(self):
        # Load in data
        data=[]
        # poynter_df=pd.read_csv(self.poynter_data_path).dropna()
        poynter_df=pd.read_csv(self.poynter_data_path).iloc[:,1:]
        cord19_df=pd.read_csv(self.cord19_data_path).iloc[:,1:]

        poynter_df=poynter_df.drop_duplicates(subset='claim', keep="first")
        # index=poynter_df.duplicated()
        # print(poynter_df[index])
        poynter=list(poynter_df["claim"]+" "+poynter_df["explanation"])
        data_urls=list(poynter_df["reference_url"].values)
        data_regions=list(poynter_df["region"])
        titles=list(poynter_df["claim"])
        content=list(poynter_df["explanation"])

        cord19_titles=list(cord19_df["title"].values)
        cord19_text=list(cord19_df["abstract"].values)

        self.sources=[]
        # Preprocess data
        preprocessed_data=[]
        self.text_t=[]
        self.titles=[]
        i=0
        for t in poynter:
            preprocessed_data.append(self.preprocess(t))
            self.text_t.append(content[i])
            self.titles.append(titles[i])
            self.sources.append("poynter")
            i+=1
        
        i=0
        for t in cord19_text:
            if (type(t)==float):
                i+=1
                continue
            preprocessed_data.append(self.preprocess(t))
            self.text_t.append(t)
            self.sources.append("cord19")
            data_urls.append(None)
            data_regions.append(None)
            self.titles.append(cord19_titles[i])
            i+=1
            
        


        data=dict(list(zip(range(len(preprocessed_data)),preprocessed_data)))
        self.url_mapping=dict(list(zip(range(len(preprocessed_data)),data_urls)))
        self.region_mapping=dict(list(zip(range(len(preprocessed_data)),data_regions)))
        self.data=data

        # self.covid_model = Word2Vec.load(self.covid_w2v_path)
        # self.all_model = Word2Vec.load(self.all_w2v_path)

        self.w2v_model=Word2Vec.load(self.covid_w2v_path)

        # initialise inverted index
        i=0
        self.inverted_index={}
        self.ids=[]
        cur_id=None
        self.doc_text={}
        self.wv=False

        # iterate over all documents
        for (d_id, d) in self.data.items():
            cur_id=d_id
            text=d            
            self.ids.append(cur_id)
            self.doc_text[cur_id]=text
            
            # update inverted index using headline and text for document "cur_id"
            self.inverted_index=self.get_inverted_index(text,cur_id)

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
        t_vec = np.sum([self.w2v_model[t] for t in terms],axis=0)/len(terms)
        for d in self.ids:
            d_text=self.doc_text[d]
            d_vec=np.zeros(100)
            i=0
            for t in d_text:
                if t not in self.w2v_model.wv.vocab:
                    continue
                d_vec=d_vec+self.w2v_model[t]
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
        
    def get_docs_with_terms(self,terms,dataset):
        docs=[]
        for t in terms:
            if t not in self.inverted_index.keys():
                continue
            for d in list(self.inverted_index[t].keys()):
                if d not in docs:
                    if self.sources[d]==dataset or dataset=="all":
                        docs.append(d)
        return docs

    def parse_tfidf_query(self,q,wv=False,dataset="poynter"):
        # if wv:
        #     self.w2v_model=w2v_model
        weighted_docs={}
        self.wv=wv
        terms=self.preprocess(q)

        if wv:
            docs=[]
            for i in self.ids:
                 if self.sources[i]==dataset or dataset=="all":
                     docs.append(i)
        else:
            docs=self.get_docs_with_terms(terms,dataset)
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
    
    def retrieve_documents(self, claim, retrieve_num=5,dataset="poynter"):
        retrieved_text=[]
        retrieved_urls=[]
        retrieved_regions=[]
        retrieved_titles=[]
        retrieved_docs=self.parse_tfidf_query(claim,dataset=dataset)
        article_ids=list(retrieved_docs)[0:retrieve_num]
        retrieved_scores=np.array(list(retrieved_docs.values()))
        if retrieved_scores!=[]:
            retrieved_scores /= np.max(retrieved_scores)
            retrieved_scores=retrieved_scores[0:retrieve_num]
        else:
            retrieved_scores=[]
        for i in article_ids:
            retrieved_text.append(self.text_t[i])
            retrieved_urls.append(self.url_mapping[i])
            retrieved_regions.append(self.region_mapping[i])
            retrieved_titles.append(self.titles[i])
        return retrieved_text, retrieved_urls, retrieved_scores, retrieved_regions,retrieved_titles
