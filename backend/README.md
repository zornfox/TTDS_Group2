# Evaluation Metrics
- Result file naming: test_id + "_results.tsv"
- test_id is an unique ID for tracking afterwards

## 03_result.tsv
- using *ii.parse_tfidf_query(claim,wv=True,w2v_model=all_model)*
- metrics:

**metric**|**depth**|**score**
:-----:|:-----:|:-----:
map|1|0.634
map|3|0.684
<ins><strong><em>map|<ins><strong><em>5|<ins><strong><em>0.691
map|10|0.697
map|20|0.699
map|all|0.701
precision|1|0.635
precision|3|0.249
precision|5|0.156
precision|10|0.082
precision|20|0.043
precision|all|0.000
reciprocal\_rank|1|0.635
reciprocal\_rank|3|0.684
reciprocal\_rank|5|0.691
reciprocal\_rank|10|0.697
reciprocal\_rank|20|0.699
reciprocal\_rank|all|0.701

## 02_result.tsv
- using *ii.parse_tfidf_query(claim,wv=True,w2v_model=covid_model)*
- metrics:

**metric**|**depth**|**score**
:-----:|:-----:|:-----:
map|1|0.637
map|3|0.684
<ins><strong><em>map|<ins><strong><em>5|<ins><strong>0.692
map|10|0.698
map|20|0.700
map|all|0.702
precision|1|0.637
precision|3|0.248
precision|5|0.156
precision|10|0.082
precision|20|0.043
precision|all|0.000
reciprocal\_rank|1|0.637
reciprocal\_rank|3|0.684
reciprocal\_rank|5|0.692
reciprocal\_rank|10|0.698
reciprocal\_rank|20|0.700
reciprocal\_rank|all|0.702

## 01_result.tsv
- using *ii.parse_tfidf_query(claim)*
- metrics:

**metric**|**depth**|**score**
:-----:|:-----:|:-----:
map|1|0.673
map|3|0.722
<ins>***map***|<ins>***5***|<ins>***0.728***
map|10|0.732
map|20|0.735
map|all|0.736
precision|1|0.674
precision|3|0.262
precision|5|0.162
precision|10|0.084
precision|20|0.044
precision|all|0.000
reciprocal\_rank|1|0.674
reciprocal\_rank|3|0.722
reciprocal\_rank|5|0.728
reciprocal\_rank|10|0.732
reciprocal\_rank|20|0.735
reciprocal\_rank|all|0.736

