python main.py --mode act \
               --im_width 128 \
               --im_height 128 \
               --im_channels 3 \
               --data_root data/variation_test/episodes \
               --random_seed 0 \
               --conv_layers_num 3 \
               --total_itr 20 \
               --state_index 7 \
               --keep_prob 0.5 \
               --norm_type layer_norm \
               --conv_weights_init random \
               --conv_filters_num '[30,30,30]' \
               --conv_filters_size '[3,3,3]' \
               --conv_strides '[[1, 2, 2, 1]-[1, 2, 2, 1]-[1, 2, 2, 1]]'\
               --fc_layers_num 2 \
               --fc_layers_size "[200, 200]" \
               --output_dim 7 \
               --loss_multiplier 50 \
               --train_lr 0.001 