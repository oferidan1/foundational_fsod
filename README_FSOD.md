#run inference with PT
python tools/test_net.py --is_gdino=1 --is_sl=1 --data_source='voc'

#create queries PT from train data

#batch rename class10_ to class7_:
rename 's/class10_/class7_/g' *.pt