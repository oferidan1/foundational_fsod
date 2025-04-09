#run inference with PT
python tools/test_net.py --is_gdino=1 --is_sl=1 --data_source='voc' --config-file='configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml'

#create queries PT from train data
python tools/test_net.py --is_gdino=1 --is_sl=0 --data_source='voc' --is_create_fs=1 --config-file='configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml'

#batch rename class0_ to class7_:

rename 's/class0_/class7_/g' *.pt
