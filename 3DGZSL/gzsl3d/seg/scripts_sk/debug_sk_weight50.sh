cd ../../../..
pip3 install -e 3DGZSL/
cd 3DGZSL/gzsl3d/seg/

# Experiment 0
python3 train_point_sk.py  --dataset="sk" --iter=10  --eval-interval=5 --epochs=20 \
--savedir="../kpconv/results/Log_SemanticKITTI/checkpoints" \
--unseen_weight=50 --checkname="sk_debug_gmmn_weighted50_zsl_debug" \
--load_embedding="glove_w2v" --w2c_size=600 --embed_dim=600 --proto_ratio 0.04 --mask_ratio 0.2
