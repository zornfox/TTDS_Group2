from backend_model import Model

# run this once
# --------
m=Model()
m.prepare_model()
# --------
print("done preparing")
# run this for any incoming claims
claim="donald trump plans social media comeback"
# Get articles from poynter by default
# articles_text, articles_urls,scores,regions=m.retrieve_documents(claim)

# Specify dataset for retrieval from poynter
articles_text, articles_urls,scores,regions=m.retrieve_documents(claim, dataset="poynter")

# Specify dataset for retrieval from research papers
# articles_text, articles_urls,scores,regions=m.retrieve_documents(claim, dataset="cord19")

# Specify dataset for retrieval from both combined
# articles_text, articles_urls,scores,regions=m.retrieve_documents(claim, dataset="all")

for i in range(5):
    text=articles_text[i]
    url=articles_urls[i]
    score=scores[i]
    region=regions[i]
    print(text,url,score,region)
    print("-"*10)
