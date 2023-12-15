cd ..
# 18, 19, 20, 15, 25, 30, 28, 27, 26, 22, 24, 23, 21, 17, 16
python3 s3dis_seg.py --test_step=1.0 --savedir="./Final_result_step/ex0_20" --cluster=True \
--use_zsl_head=True --use_no_attribute=True --test --zsl_trained_path="../../../seg/run/s3dis/s3dis_gmmn_weighted50_zsl_debug/experiment_0/20_model.pth.tar" \
--bias=0.4
python3 s3dis_eval.py --p="Final_result_step/ex0_20" --area=1