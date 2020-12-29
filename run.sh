import tensorflow.compat.v1 as tf
for e in tf.train.summary_iterator('events.out.tfevents.1608614125.VM-0-9-centos'):
    for v in e.summary.value:
        if 'loss' not in v.tag:
            print(v.tag)
            print(v.simple_value)
index/hit
69.03646087646484
index/mrr
29.37898063659668
index/hit 69.08118358438584 4
index/mrr 29.594031914908314 4


import tensorflow.compat.v1 as tf
for e in tf.train.summary_iterator('events.out.tfevents.1608614367.VM-0-9-centos'):
    for v in e.summary.value:
        if 'loss' not in v.tag:
            print(v.tag)
            print(v.simple_value)
index/hit
50.81172561645508
index/mrr
17.383739471435547

python preprocess.py --dataset=diginetica
python preprocess.py --dataset=yoochoose

CUDA_VISIBLE_DEVICES=0 python main.py --dataset=diginetica
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=yoochoose1_64
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=yoochoose1_4

CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --dataset=yoochoose1_64 --gpus=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --dataset=diginetica --gpus=4
