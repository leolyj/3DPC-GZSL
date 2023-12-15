cd ../../../..
pip3 install -e 3DGZSL/
cd 3DGZSL/gzsl3d/seg/

##Experiment 0
python3 train_point_s3dis.py --bias=0.4 --iter=1000 --epochs=20 --eval_interval=5 \
--savedir="../convpoint/examples/s3dis/ConvPoint_s3dis_ZSL4" \
--rootdir="../../data/s3dis/processed_data" --cluster=True --temperature 1.0 --unseen_weight=50 --checkname="s3dis_gmmn_weighted50_zsl_retrained" \
--load_embedding="glove_w2v" --w2c_size=600 --embed_dim=600  --proto_ratio 0.04 --mask_ratio 0.2
