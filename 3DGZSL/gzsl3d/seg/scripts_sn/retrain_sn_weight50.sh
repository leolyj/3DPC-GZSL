cd ../../../..
pip3 install -e 3DGZSL/
cd 3DGZSL/gzsl3d/seg/

# Experiment 0
python3 train_point_sn.py --iter=300 --epochs=20 --eval-interval=5 \
--config_savedir="../fkaconv/examples/scannet/FKAConv_scannet_ZSL4" \
--rootdir="../../data/scannet/"  --unseen_weight=50 --checkname="sn_gmmn_weighted50_zsl_retrained" \
--load_embedding="glove_w2v" --w2c_size=600 --embed_dim=600 --proto_ratio 0.04 --mask_ratio 0.2





