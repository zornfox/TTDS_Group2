from backend_model import Model

# run this once
# --------
m=Model()
m.prepare_model()
# --------

# run this for any incoming claims
claim="coronavirus is not a virus but a bacteria"
articles=m.retrieve_documents(claim)

for x in articles[0:5]:
    print(x)
    print("-"*10)