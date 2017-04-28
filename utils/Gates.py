from inputdata import *

class Dataset:
	def __init__(self, x, y):
		self.X = x
		self.Y = y
		self.count = 0
		self.num = len(self.Y)
	def stack(self):
		self.Y = np.hstack(self.Y)
		self.Y = np.float32(np.eye(7)[self.Y.astype(int)]) 
	def batch(self, batch_size):
		if self.count+batch_size>=self.num:
			self.X, self.Y = randomize2(self.X, self.Y)
			self.count = 0
		self.count+=batch_size
		return self.X[self.count-batch_size:self.count], self.Y[self.count-batch_size:self.count]



class Gates:

	def test_set(self, test_folders, picklename, image_size, n, mask=False):
		test_name=sorted(os.listdir(test_folders))
		test_dataset = load_letter(test_folders, n, image_size)
		test_dataset = test_dataset/255.0
		label=[0,0,0,0,0,0,0,n]
		for j in range(6):
			label[j+1]= find_label(test_name,test_name[label[j]][0:3], label[j])
			test_label=np.zeros(n)
			for i in range(7):
			    test_label[label[i]:label[i+1]]=i
		print("test_data")
		if mask:
			np.putmask(test_dataset, test_dataset < 0.35, 0.0)
			np.putmask(test_dataset, test_dataset >= 0.35, 1.0)
		save={'X':test_dataset, 'Y':test_label}
		file_pickle(picklename, save, force=True)

	def train_set(self, folders, picklename, image_size, mask=False):	
		train_folders = [os.path.join(folders, d)
			for d in sorted(os.listdir(folders))
			if os.path.isdir(os.path.join(folders, d))]
		train_dataset, train_labels = maybe_load(train_folders, image_size, force=True)
		train_dataset = train_dataset/255.0
		print("train_data")
		if mask:
			np.putmask(train_dataset, train_dataset < 0.35, 0.0)
			np.putmask(train_dataset, train_dataset >= 0.35, 1.0)
		save={'X':train_dataset, 'Y':train_labels}
		file_pickle(picklename, save, force=True)

		


	def __init__(self, image_size=28, filename="gates_data.tar.gz", 
		force= False, inputsize1=5000, inputsize2=0, mask = True):

		data_folders = maybe_extract(filename, force=False)
		pickle_names = [] 

		for i,folder in enumerate(data_folders):
			
			picklename = folder+".pickle"
			pickle_names.append(picklename)
			
			if not os.path.isdir(picklename) or force:
				
				if i==0:
					self.test_set(folder, picklename, image_size, 10000, mask = mask)
				elif i==1:
					self.test_set(folder, picklename, image_size, 1596)
				elif i==2:
					self.train_set(folder, picklename, image_size, mask = mask)
				else:
					self.train_set(folder, picklename, image_size)
		

		
		with open(pickle_names[2], 'rb') as f:
			train1 = pickle.load(f)


		with open(pickle_names[3], 'rb') as f:
			train2 = pickle.load(f)

		train_dataset, train_labels = make_arrays((inputsize1+inputsize2)*7, image_size)		
		maxsize1 = len(train1['Y'])/7
		maxsize2 = len(train2['Y'])/7
		size     = inputsize1+inputsize2
		for i in range(7):
			s1 = i*maxsize1
			s2 = i*maxsize2
			s  = i*size
			train_dataset[s:s+inputsize1] = train1['X'][s1:s1+inputsize1]
			train_dataset[s+inputsize1:s+size] = train2['X'][s2:s2+inputsize2]
			train_labels[s:s+inputsize1] = train1['Y'][s1:s1+inputsize1]
			train_labels[s+inputsize1:s+size] = train2['Y'][s2:s2+inputsize2]




		with open(pickle_names[0], 'rb') as f:
			test = pickle.load(f)

		
		with open(pickle_names[1], 'rb') as f:
			test2 = pickle.load(f)
        
       		
		'''
		self.AND  = train['X'][    0: 5000]
		self.NAND = train['X'][ 5000:10000]
		self.NOR  = train['X'][10000:15000]
		self.NOT  = train['X'][15000:20000]
		self.OR   = train['X'][20000:25000]
		self.XNOR = train['X'][25000:30000]
		self.XOR  = train['X'][30000:35000]
		'''
		#random the dataset

		train_x, train_y = randomize(train_dataset, train_labels)    
		self.train = Dataset(train_x, train_y)
		self.test  = Dataset(test['X'], test['Y'])
		self.valid = Dataset(test2['X'], test2['Y'])
		#self.new_test  = Dataset(new_test['X'], new_test['Y'])









