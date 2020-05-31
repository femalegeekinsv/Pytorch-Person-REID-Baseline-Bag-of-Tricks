import torch
from tools import CatMeter, cosine_dist, visualize_ranked_results
import time
import glob

def inference_buildgallery(config, base, loaders):

	base.set_eval()
				
	dir_q = '/home/teresa/reID/temp'
	q_list = sorted(glob.glob(dir_q + '/*'))
	
	for i, q in enumerate(q_list):
		print('query', i, q)
		loaders.set_query(q, 0, 1) #TODO: mod for multiple queries
		loaders.process_query_data()
		features_g = None
		
		with torch.no_grad():	
			for batch_num, data in enumerate(loaders.query_loader):			
				images, pids, cids = data			
				features_q = base.model(images) #can be multiple features				
				if len(loaders.gallery) > 0:
					for g in loaders.gallery:
						path, feat, pid, cid = g
						if features_g is None:
							features_g = feat.unsqueeze(0)
						else:
							features_g = torch.cat([features_g, feat.unsqueeze(0)], dim=0)
					#extract feat from tuples and convert to torch
					#print(features_g)
					distance = cosine_dist(features_q, features_g).data.cpu().numpy()
					print(distance)
					indices = np.argsort(distance, axis=1)[:,::-1] #smallest distance first, reverse order when add to gallery
					print(indices)
					closest_dist = distance[0][indices[0][0]]
					print('closest match with dist',loaders.gallery[indices[0][0]][0], distance[0][indices[0][0]])
					if closest_dist > 0.6:
						pids = [loaders.gallery[indices[0][0]][2]] #2 for pid
					else:
						pids = [i]
					#pids = assign_pid(distance,loaders.gallery) #one query has one row of distance hence 
					
					#another function to sort with index and assign PID
				#for f in features_q: #TODO: mod for muliple queries
				else:
					pids = [i] #first pid for empty gallery
				for fp in zip(features_q,pids):
					loaders.add_gallery_all(q, fp[0], fp[1], 1)	
	print(loaders.gallery)

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
