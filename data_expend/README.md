# Data_amplification
1. -> open lable program --TrainDIR (img folder) --SaveDIR (save folder .txt) 
python3 labeling_locatev2.py --TrainDIR origin_img/img/blue_bottle/ --SaveDIR origin_data/
2. -> add image_path,target_x,target_y,target_angle,ix,iy,rx,ry on top of origin data
3. -> use check_data.py show locate img example: python3 check_data.py --TrainDIR /origin_data/blackbox_2020-10-23_15_08_17_.csv
4. -> use data_expand.py expand data exanple: python3 data_expand.py --expand_time 3



