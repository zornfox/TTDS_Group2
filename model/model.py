from os.path import exists
import re
import pandas as pd
import numpy as np
import pickle

#url
#source
#region
#score
#

class Model():
    def __init__(self,stop_word_path,poynter_data_path,cord19_data_path,save_path):
        self.stop_word_path = stop_word_path
        self.poynter_data_path = poynter_data_path
        self.cord19_data_path = cord19_data_path
        self.save_path = save_path

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
        # initialise inverted index
        self.inverted_index = {}
        self.ids = []
        self.text_t = []
        self.sources = []
        self.url_mapping = {}
        self.region_mapping = {}
        self.titles=[]

        if exists(self.save_path):
            with open( self.save_path, 'rb') as f:
                self.inverted_index, self.ids, self.text_t, self.sources,self.url_mapping,self.region_mapping,self.titles = pickle.load(f)
        else:

            poynter_df = pd.read_csv(self.poynter_data_path).dropna().iloc[:, 1:]
            cord19_df = pd.read_csv(self.cord19_data_path).dropna().iloc[:, 1:]
            # Load in data
            data=[]

            poynter_df=poynter_df.drop_duplicates(subset='content', keep="first")
            poynter=list(poynter_df["content"]+" "+poynter_df["explanation"])
            data_urls=list(poynter_df["reference_url"].values)
            data_regions=list(poynter_df["region"])
            titles=list(poynter_df["content"])
            content=list(poynter_df["explanation"])
            cord19_titles=list(cord19_df["title"].values)
            cord19_text=list(cord19_df["abstract"].values)
            
            # Preprocess data
            preprocessed_data=[]


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



            # iterate over all documents
            for (d_id, d) in self.data.items():
                self.ids.append(d_id)
                # update inverted index using headline and text for document "cur_id"
                self.inverted_index = self.get_inverted_index(d, d_id)

            with open( self.save_path, 'wb') as f:
                pickle.dump((self.inverted_index, self.ids, self.text_t,self.sources,self.url_mapping,self.region_mapping,self.titles), f)


    
    def TFIDF(self,t,d):
        N=len(self.ids)
        tft=np.log10(self.inverted_index[t][d][0])
        dft=len(self.inverted_index[t].keys())
        ldft=np.log10(N/dft)
        return (1+tft)*ldft
        
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
        weighted_docs = {}
        terms = self.preprocess(q)
        docs = self.get_docs_with_terms(terms, dataset)
        for d in docs:
            cur_w=0
            for t in terms:
                if t not in self.inverted_index.keys():
                    continue
                # if term not in this docuemnt
                if self.inverted_index[t].get(d)==None:
                    continue
                else:
                    cur_w = cur_w + self.TFIDF(t, d)
            weighted_docs[d]=cur_w
        sorted_docs = {k: v for k, v in sorted(weighted_docs.items(), key=lambda item: item[1], reverse = True)}
        return sorted_docs
    
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
            retrieved_scores=1
        for i in article_ids:
            retrieved_text.append(self.text_t[i])
            retrieved_urls.append(self.url_mapping[i])
            retrieved_regions.append(self.region_mapping[i])
            retrieved_titles.append(self.titles[i])
        return retrieved_text, retrieved_urls, retrieved_scores, retrieved_regions,retrieved_titles
