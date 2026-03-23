mkdir sample
cd sample
mkdir 20220101
cd ../
python generate_tfrecord.py -n 1000 -o sample/20220101/data.tfrecord --label_pos_prob 0.3
