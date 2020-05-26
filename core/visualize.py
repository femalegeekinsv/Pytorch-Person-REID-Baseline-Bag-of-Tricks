import torch
from tools import CatMeter, cosine_dist, visualize_ranked_results
import time
import glob

def visualize(config, base, loaders):

	base.set_eval()

	dir_q = '/home/teresa/reID/output_filter_2019-08-04-15-45-ip-101.flv/'
	q_list = glob.glob(dir_q + '/*')
	for i, q in enumerate(q_list):
		loaders.add_query(q, 0, 1)
		print(i, q)
	#	if i>100:
	#		break
	dir_g = '/home/teresa/reID/output_filter_2019-08-04-15-45-ip-102.flv/'
	g_list = glob.glob(dir_g + '/*')
	for i, g in enumerate(g_list):
		loaders.add_gallery(g, 0, 1)
		print(i, g)
	dir_g = '/home/teresa/reID/output_filter_2019-08-04-15-45-ip-103.flv/'
	g_list = glob.glob(dir_g + '/*')
	for i, g in enumerate(g_list):
		loaders.add_gallery(g, 0, 1)
		print(i, g)
	dir_g = '/home/teresa/reID/output_filter_2019-08-04-15-45-ip-104.flv/'
	g_list = glob.glob(dir_g + '/*')
	for i, g in enumerate(g_list):
		loaders.add_gallery(g, 0, 1)
		print(i, g)				
	
	#call this when done adding
	loaders.process_data()
	
	# meters
	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

	# init dataset
	if config.visualize_dataset == 'market':
		_datasets = [loaders.market_query_samples.samples, loaders.market_gallery_samples.samples]
		_loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
	elif config.visualize_dataset == 'duke':
		_datasets = [loaders.duke_query_samples.samples, loaders.duke_gallery_samples.samples]
		_loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
	elif config.visualize_dataset == 'customed':
		_datasets = [loaders.query_samples, loaders.gallery_samples]
		_loaders = [loaders.query_loader, loaders.gallery_loader]

	print('query and then gallery',_datasets)
	# compute query and gallery features
	with torch.no_grad():
		for loader_id, loader in enumerate(_loaders):
			for batch_num, data in enumerate(loader):
				# compute feautres
				images, pids, cids = data
				print('image len',len(images))
				#print('loading all query followed by gallery images',len(pids), batch_num)
				start_time = time.time()
				features = base.model(images)
				print('1 time get features',time.time()-start_time)
				start_time = time.time()
				features = base.model(images)
				print('2 time get features',time.time()-start_time)
				start_time = time.time()
				features = base.model(images)
				print('3 time get features',time.time()-start_time)	
				start_time = time.time()
				features = base.model(images)
				print('4 time get features',time.time()-start_time)
				start_time = time.time()
				features = base.model(images)
				print('5 time get features',time.time()-start_time)
				start_time = time.time()
				features = base.model(images)
				print('6 time get features',time.time()-start_time)												
				# save as query features
				if loader_id == 0:
					query_features_meter.update(features.data)
					query_pids_meter.update(pids)
					query_cids_meter.update(cids)
				# save as gallery features
				elif loader_id == 1:
					gallery_features_meter.update(features.data)
					gallery_pids_meter.update(pids)
					gallery_cids_meter.update(cids)

		img_path = ''

	# compute distance
	query_features = query_features_meter.get_val()
	gallery_features = gallery_features_meter.get_val()
	start_time = time.time()
	distance = cosine_dist(query_features, gallery_features).data.cpu().numpy()
	print('time get dist',time.time()-start_time)
	print(len(distance),len(distance),type(distance))
	print('dist size',len(distance),len(distance[0]))

	# visualize
	visualize_ranked_results(distance, _datasets, config.visualize_output_path, mode=config.visualize_mode, topk=20)
