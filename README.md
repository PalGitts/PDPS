# Exposing Long-Tail Safety Failures in Large Language Models through Efficient Diverse Response Sampling

Accepted by <b>Transactions on Machine Learning Research (TMLR)</b>


Reviewed on OpenReview: https://openreview.net/forum?id=tHfAskovWI


![INFORM Framework Image](images/PDPS_mechanism.png)


## Setup

Install all dependencies using the following command:

```bash
$ pip install -r requirements.txt
```

**Note**: Use transformers==4.55.4 for **Diverse Beam Search**.

## Instructions

1. pdps_attack.py : It initiates the Progressive Diverse Population Sampling (PDPS). Use, 

```bash
CUDA_VISIBLE_DEVICES=0 python3 pdps_attack.py --model qwen7b_inst --dataset A --task_id 0 --temperature 1.0 --top_p 1.0 --hyp_lambda 64.0
```

2. 
