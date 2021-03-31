# TTDS_Group2

 
## 1 git clone --branch newMain https://github.com/zornfox/TTDS_Group2.git
## 2 cd TTDS_Group2
## 3 python3 -m venv env
## 4 source env/bin/activate
## 5 pip3 install -r requirements.txt
## 6 python3 server.py
## 7 http://127.0.0.1:5000/

```bash

## Build Command
docker build -t gitdorker .

## Basic Run Command
docker run -it gitdorker

## Run Command
docker run -it -v $(pwd)/tf:/tf gitdorker -tf tf/TOKENSFILE -q tesla.com -d dorks/DORKFILE -o tesla

## Run Command
docker run -it -v $(pwd)/tf:/tf xshuden/gitdorker -tf tf/TOKENSFILE -q tesla.com -d dorks/DORKFILE -o tesla

```

