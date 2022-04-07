# Keyphrase Generation (built on OpenNMT-py)

## Prerequisites

- You need `Python3.6` to run the code. The list of the dependencies are in `requrements.txt`. Run:

  ```
  python -m pip install -r requirements.txt
  ```

- After the installation of geoopt, replace the math.py in ../geoopt/manifolds/stereographic by the math.py in out project

### Train a One2Seq model

```
python train.py -config config/train/config-rnn-keyphrase-one2seq-diverse.yml
```



### Train a One2One model

```
python train.py -config config-rnn-keyphrase-one2one-kp20k-hyperbolic_network_attention
```

### Run generation and evaluation

```
python kp_gen_eval.py -tasks pred eval report -config config/test/config-test-keyphrase-one2seq.yml -data_dir data/keyphrase/meng17/ -ckpt_dir models/keyphrase/meng17-one2seq-kp20k-topmodels/ -output_dir output/meng17-one2seq-topbeam-selfterminating/meng17-one2many-beam10-maxlen40/ -testsets duc inspec semeval krapivin nus -gpu -1 --verbose --beam_size 10 --batch_size 32 --max_length 40 --onepass --beam_terminate topbeam --eval_topbeam
```

