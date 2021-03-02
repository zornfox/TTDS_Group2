# Evaluation Metrics
- Result file naming: test_id + "_results.tsv"
- test_id is an unique ID for tracking afterwards

## 03_result.tsv
- using *ii.parse_tfidf_query(claim,wv=True,w2v_model=all_model)*
- metrics:

**metric**|**depth**|**score**
:-----:|:-----:|:-----:
map|1|0.000
map|3|0.000
map|5|0.000
map|10|0.000
map|20|0.000
map|all|0.032
precision|1|0.000
precision|3|0.000
precision|5|0.000
precision|10|0.000
precision|20|0.000
precision|all|0.000
reciprocal\_rank|1|0.000
reciprocal\_rank|3|0.000
reciprocal\_rank|5|0.000
reciprocal\_rank|10|0.000
reciprocal\_rank|20|0.000
**reciprocal\_rank**|**all**|**0.032**

## 02_result.tsv
- using *ii.parse_tfidf_query(claim,wv=True,w2v_model=covid_model)*
- metrics:

**metric**|**depth**|**score**
:-----:|:-----:|:-----:
map|1|0.000
map|3|0.000
map|5|0.000
map|10|0.000
map|20|0.000
map|all|0.032
precision|1|0.000
precision|3|0.000
precision|5|0.000
precision|10|0.000
precision|20|0.000
precision|all|0.000
reciprocal\_rank|1|0.000
reciprocal\_rank|3|0.000
reciprocal\_rank|5|0.000
reciprocal\_rank|10|0.000
reciprocal\_rank|20|0.000
**reciprocal\_rank**|**all**|**0.032**

## 01_result.tsv
- using *ii.parse_tfidf_query(claim)*
- metrics:

**metric**|**depth**|**score**
:-----:|:-----:|:-----:
map|1|0.000
map|3|0.000
map|5|0.000
map|10|0.000
map|20|0.001
map|all|0.033
precision|1|0.000
precision|3|0.000
precision|5|0.000
precision|10|0.000
precision|20|0.001
precision|all|0.000
reciprocal\_rank|1|0.000
reciprocal\_rank|3|0.000
reciprocal\_rank|5|0.000
reciprocal\_rank|10|0.000
reciprocal\_rank|20|0.001
**reciprocal\_rank**|**all**|**0.033**

