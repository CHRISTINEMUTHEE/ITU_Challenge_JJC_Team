# Thor S1/2 exps

Run using Segmenter's LinearDecoder. Cross-entropy used for segmentation and MSE for height regression. Building and water segmentation are weighted (0.3, 0.7) for actual negative and actual positive respectively.

Pass multiple both thor s1 and s2 dir paths to `--train-embeddings-dir` to train on both, concatenated on the channel dimension. Pass one to train on one. 
```shell 
python train.py \
    --model-type linear_decoder \
    --loss-type simple_integrated_loss \
    --train-embeddings-dir <path to>/thor_s1_emb <path to>/thor_s2_emb \
    --train-targets-dir /home/admin/john/data/embed2heights/embed2heights/data/train/labels \
    --experiment-name s1s2_concat \
    --epochs 30 --batch-size 32 --patch-size 256
```