'''
given a pre-trained model, visualize a customed dataset
pre-trained model can be got by running main.py in a train mode
'''

import argparse
from core import CustomedLoaders, OnTheFlyLoaders, DemoBase, visualize
from tools import make_dirs


def demo(config):
    # init loaders and base
    #loaders = CustomedLoaders(config)
    loaders = OnTheFlyLoaders(config)
    base = DemoBase(config)

    # visualization
    base.resume_from_model(config.resume_visualize_model)
    make_dirs(config.visualize_output_path)
    visualize(config, base, loaders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model configuration
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128], help='should be consistent with pre-trained model')
    parser.add_argument('--pid_num', type=int, default=751, help='751 for Market-1501, 702 for DukeMTMC-reID')

    # demo configuration
    parser.add_argument('--visualize_dataset', type=str, default='customed', help='please do not change')
    parser.add_argument('--visualize_output_path', type=str, default='results/visualization/', help='path to save visualization results, only availiable under visualize model')
    parser.add_argument('--query_path', type=str, default='path/to/customed/query/images/', help='customed query image path')
    parser.add_argument('--gallery_path', type=str, default='path/to/customed/gallery/images/', help='customed gallery image path')
    parser.add_argument('--resume_visualize_model', type=str, default='/path/to/pretrained/model.pkl', help='path to pretrained model.pkl')
    parser.add_argument('--visualize_mode', type=str, default='all', help='intra-camera, inter-camera, all')

    config = parser.parse_args()
    demo(config)
    
    
    #python main.py --mode train --train_dataset market --test_dataset market --market_path /home/teresa/reID/reid-strong-baseline/data/market1501 --output_path ./results/market/

    #python main.py --mode test --train_dataset market --test_dataset market --market_path /home/teresa/reID/reid-strong-baseline/data/market1501/ --resume_test_model results/market/model_120.pkl --output_path ./results/test-on-market/

    #python main.py --mode visualize --visualize_mode inter-camera --train_dataset market --visualize_dataset market --market_path /home/teresa/reID/reid-strong-baseline/data/market1501/ --resume_visualize_model ./results/market/model_120.pkl --visualize_output_path ./results/vis-on-market/

    #python demo.py --resume_visualize_model ./results/market/model_120.pkl --visualize_output_path ./results/vis-on-cus  --query_path ./../cus/query/ --gallery_path ./../cus/gallery/
