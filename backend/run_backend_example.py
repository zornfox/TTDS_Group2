from backend_model import Model

# run this once
# --------
m=Model()
m.prepare_model()
# --------
print("done preparing")
# run this for any incoming claims
claim="coronavirus is not a virus but a bacteria"
articles_text, articles_urls,scores=m.retrieve_documents(claim)

for i in range(5):
    text=articles_text[i]
    url=articles_urls[i]
    score=scores[i]
    print(text,url,score)
    print("-"*10)
