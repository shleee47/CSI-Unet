#python main.py --backbone resnet50 --ytvos_path data --masks
python main.py --backbone resnet101 --ytvos_path data --masks --pretrained_weights ./models/pretrained/vistr_r101.pth --output_dir ./checkpoint/resnet101_e1_d1/
