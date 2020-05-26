import argparse
import os
import ast
from core import ReIDLoaders, Base, train_an_epoch, test, visualize
from tools import make_dirs, Logger, os_walk, time_now


def main(config):

	# init loaders and base
	loaders = ReIDLoaders(config)
	base = Base(config)

	# make directions
	make_dirs(base.output_path)

	# init logger
	logger = Logger(os.path.join(config.output_path, 'log.txt'))
	logger(config)


	assert config.mode in ['train', 'test', 'visualize']
	if config.mode == 'train':  # train mode

		# automatically resume model from the latest one
		if config.auto_resume_training_from_lastest_steps:
			print('resume', base.output_path)
			start_train_epoch = base.resume_last_model()
		#start_train_epoch = 0

		# main loop
		for current_epoch in range(start_train_epoch, config.total_train_epochs+1):
			# save model
			base.save_model(current_epoch)
			# train
			base.lr_scheduler.step(current_epoch)
			_, results = train_an_epoch(config, base, loaders)
			logger('Time: {};  Epoch: {};  {}'.format(time_now(), current_epoch, results))

		# test
		base.save_model(config.total_train_epochs)
		mAP, CMC = test(config, base, loaders)
		logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {}'.format(time_now(), config.test_dataset, mAP, CMC))


	elif config.mode == 'test':	# test mode
		base.resume_from_model(config.resume_test_model)
		mAP, CMC = test(config, base, loaders)
		logger('Time: {}; Test Dataset: {}, \nmAP: {} \nRank: {} with len {}'.format(time_now(), config.test_dataset, mAP, CMC, len(CMC)))


	elif config.mode == 'visualize': # visualization mode
		base.resume_from_model(config.resume_visualize_model)
		visualize(config, base, loaders)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('--cuda', type=str, default='cuda')
	parser.add_argument('--mode', type=str, default='train', help='train, test or visualize')
	parser.add_argument('--output_path', type=str, default='results/', help='path to save related informations')

	# dataset configuration
	parser.add_argument('--market_path', type=str, default='/home/teresa/reID/reid-strong-baseline/data/market1501')
	parser.add_argument('--duke_path', type=str, default='/home/wangguanan/datasets/PersonReIDDatasets/Duke/occlude_DukeMTMC-reID/')
	parser.add_argument('--train_dataset', type=str, default='market', help='market, duke')
	parser.add_argument('--test_dataset', type=str, default='market', help='market, duke')
	parser.add_argument('--image_size', type=int, nargs='+', default=[256, 128])
	parser.add_argument('--p', type=int, default=16, help='person count in a batch')
	parser.add_argument('--k', type=int, default=4, help='images count of a person in a batch')

	# model configuration
	parser.add_argument('--pid_num', type=int, default=751, help='751 for Market-1501, 702 for DukeMTMC-reID')
	parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')

	# train configuration
	parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
	parser.add_argument('--base_learning_rate', type=float, default=0.00035)
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--total_train_epochs', type=int, default=120)
	parser.add_argument('--auto_resume_training_from_lastest_steps', type=ast.literal_eval, default=True)
	parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')

	# test configuration
	parser.add_argument('--resume_test_model', type=str, default='/path/to/pretrained/model.pkl', help='')
	parser.add_argument('--test_mode', type=str, default='inter-camera', help='inter-camera, intra-camera, all')

	# visualization configuration
	parser.add_argument('--resume_visualize_model', type=str, default='./results/market/model_120.pkl',
						help='only availiable under visualize model')
	parser.add_argument('--visualize_dataset', type=str, default='',
						help='market, duke, only  only availiable under visualize model')
	parser.add_argument('--visualize_mode', type=str, default='inter-camera',
						help='inter-camera, intra-camera, all, only availiable under visualize model')
	parser.add_argument('--visualize_output_path', type=str, default='results/visualization/',
						help='path to save visualization results, only availiable under visualize model')


	# main
	config = parser.parse_args()
	main(config)

   
#    python main.py --mode train --train_dataset market --test_dataset market --market_path /home/teresa/reID/reid-strong-baseline/data/market1501 --output_path ./results/market/

#    python main.py --mode test --train_dataset market --test_dataset market --market_path /home/teresa/reID/reid-strong-baseline/data/market1501/ --resume_test_model results/market/model_120.pkl --output_path ./results/test-on-market/

#    python main.py --mode visualize --visualize_mode inter-camera --train_dataset market --visualize_dataset market --market_path /home/teresa/reID/reid-strong-baseline/data/market1501/ --resume_visualize_model ./results/market/model_120.pkl --visualize_output_path ./results/vis-on-market/

#    python demo.py --resume_visualize_model ./results/market/model_120.pkl --visualize_output_path ./results/vis-on-cus  --query_path ./../cus/query/ --gallery_path ./../cus/gallery/
    
    



