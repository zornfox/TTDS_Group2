#pip install --upgrade google-cloud-storage
# Imports the Google Cloud client library
from google.cloud import storage


import pickle
from google.cloud.storage import Blob
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key/coviddoc-439bbe4bbf8f.json"
# Instantiates a client
storage_client = storage.Client()
# The name for the new bucket
bucket_name = "coviddocs"
# Creates the new bucket
bucket = storage_client.bucket(bucket_name)
filename = "model.pickle"
print("Bucket {} created.".format(bucket.name))
blob= bucket.blob(filename)


inverted_index = {}
ids = []
text_t = []
sources = []
url_mapping = {}
region_mapping = {}
titles=[]
doc_text={}
wv=False
if(blob!=None):
    with open(blob,"rb") as f:
        inverted_index, ids, text_t, sources,url_mapping,region_mapping,titles,doc_text,wv= pickle.load(f)
else:
    print("pickle is not found, create now")
    
    my_modified_dictionary=(inverted_index, ids, text_t, sources,url_mapping,region_mapping,titles,doc_text,wv)
    # blob.upload_from_file(pickle_out)
    pickle_out = pickle.dumps(my_modified_dictionary)
    blob.upload_from_string(pickle_out)
