import pdb
from recolor_utils import *
import matplotlib.pyplot as plt
import time
from threading import Thread
from numba import jit
class Palette:
	def __init__(self,img,K=5):
		
		self.bin_num = 16
		self.channels = 4
		self.img = img
		self.orig_img = img[:,:,0:3]
		self.K = K
		self.inf = float("inf")
		bin_num = self.bin_num
		self.bin_color = np.zeros((bin_num,bin_num,bin_num,3))
		self.bin_count = np.zeros((bin_num,bin_num,bin_num,1))
		self.bin_tag   = np.zeros((bin_num,bin_num,bin_num,1)) - 1  #initialize with -1
		#print("img[500][277]={}".format(img[500][285]))
		#(235,128,104)
		##initialize bin  color/lab, count/weight, tag/i-cluster
		'''
		self.data, self.norm_data,self.bin_base,self.bin_range = self.normalize(img)
		for i in range(self.bin_num):
			for j in range(self.bin_num):
				for k in range(self.bin_num):
					clr_r = self.norm2orig((i+0.5)*(1/self.bin_num),'r')
					clr_g = self.norm2orig((j+0.5)*(1/self.bin_num),'g')
					clr_b = self.norm2orig((k+0.5)*(1/self.bin_num),'b')
					self.bin_color[i,j,k] = rgb2lab([clr_r,clr_g,clr_b])

		img_data = self.data.copy();		
		for i in range(img_data.shape[0]):
			r,g,b = img_data[i]
			ri,gi,bi = int(r*self.bin_num),int(g*self.bin_num),int(b*self.bin_num)
			ri -= (ri==self.bin_num)
			gi -= (gi==self.bin_num)
			bi -= (bi==self.bin_num)

			self.bin_count[ri][gi][bi] += 1'''

		#########
		## another way to initalize(withount normalize)
		'''
		for i in range(self.bin_num):
			for j in range(self.bin_num):
				for k in range(self.bin_num):
					clr_r = (i+0.5)*self.bin_num
					clr_g = (j+0.5)*self.bin_num
					clr_b = (k+0.5)*self.bin_num
					self.bin_color[i,j,k] = rgb2lab([clr_r,clr_g,clr_b])


		img_data = np.array(self.img).reshape(-1,3);		
		for i in range(img_data.shape[0]):
			r,g,b = img_data[i]
			ri,gi,bi = math.floor(r/self.bin_num),math.floor(g/self.bin_num),math.floor(b/self.bin_num)
			ri -= (ri==self.bin_num)
			gi -= (gi==self.bin_num)
			bi -= (bi==self.bin_num)
			self.bin_count[ri][gi][bi] += 1
		print(self.bin_color)
		print(self.bin_color.shape)
		'''
		#idx = (np.arange(self.bin_num)+0.5)*(256/self.bin_num)
		#idx_r = np.repeat(np.repeat(idx.reshape(self.bin_num,1,1,1),self.bin_num,axis=1),self.bin_num,axis=2)
		#idx_g = np.repeat(np.repeat(idx.reshape(1,self.bin_num,1,1),self.bin_num,axis=0),self.bin_num,axis=2)
		#idx_b = np.repeat(np.repeat(idx.reshape(1,1,self.bin_num,1),self.bin_num,axis=0),self.bin_num,axis=1)
		#self.bin_color = np.concatenate((idx_r,idx_g,idx_b),axis=3)
		
		#41.25595568978259,-13.430644963549721 18.325524758002672
		#99.86456269387023 -0.3109940638383857 -0.08613797071691096
		#43.591213116910644 8.309873936857255 -30.38691603072777
		#57.229503813615054 30.26064995100793 31.614862560671707
		#45.24416783631266 -22.90240052940612 30.26613777060414
		#bin_num = 
		self.bin_color = np.zeros((self.bin_num,self.bin_num,self.bin_num,3))
		bin_step = 255.0/(self.bin_num-1)
		
		img_data = np.array(self.img).reshape(-1,3);		
		idx = np.array(np.around(img_data/bin_step),np.int_)
		#time1 = time.time()


		for i in range(img_data.shape[0]):
			#r,g,b = img_data[i]
			#ri,gi,bi = int(round(r/bin_step)),int(round(g/bin_step)),int(round(b/bin_step))
			ri,gi,bi = idx[i]
			self.bin_count[ri,gi,bi] += 1
			self.bin_color[ri,gi,bi] += rgb2lab(img_data[i])#rgb2lab(self.bin_color[i,j,k])
			
			
			'''
		img_data = np.array(self.img).reshape(-1,3)
		idx = np.array(np.around(img_data/bin_step),np.int_)
		r_idx = idx[:,0]
		g_idx = idx[:,1]
		b_idx = idx[:,2]
		self.bin_count[r_idx,g_idx,b_idx] += 1
		self.bin_color[r_idx,g_idx,b_idx] += rgb2lab(img_data[r,g,b])
		'''
		#time6 = time.time()
		for i in range(self.bin_num):
			for j in range(self.bin_num):
				for k in range(self.bin_num):
					if self.bin_count[i][j][k]!=0:
						self.bin_color[i,j,k] = self.bin_color[i,j,k]/self.bin_count[i,j,k]

		centers_rgb = lab2rgb4arr(self.kmeans(lock_black=True))
		#time2 = time.time()
		self.centers_lab = rgb2lab4arr(centers_rgb)
		self.orig_centers = self.centers_lab.copy()
		self.user_edited = self.centers_lab.copy()
		self.origL = np.append(np.insert(self.orig_centers[:,0],0,0),100)

		#compute params-sigma
		sigma = getSigma(self.orig_centers)

		#compute param-phi
		def phi(r):
			#compute params phi(r) = exp(-(r^2)/sigma^2)
			return math.exp(-r*r/(2*sigma*sigma))
				
		equation_group = np.zeros((self.K,self.K))
		for i in range(self.K):
			for j in range(self.K):
				equation_group[i][j] = phi(labDistance(self.orig_centers[j],self.orig_centers[i]))
		
		lambda_ij = getLambda(equation_group,np.eye(self.K))
		#print("equation_group={}".format(equation_group))
		#print("lambda_ij={}".format(lambda_ij))
	
		#precompute color(Lab) shift in grid. Then use trilinear interpolate to compute the rest colors(rgb)
		g = 12 #grid size 
		self.g = g
		self.wi = np.zeros((g,g,g,self.K)) #weights of RBF functions:how much effection each center/cluster cause to each grid 
		gsize = (math.ceil(256/(g-1))) #24
		self.gsize = gsize
		self.g_lab = np.zeros((g,g,g,3))
		self.g_rgb = np.zeros((g,g,g,3))
		for ii in range(g):
			for jj in range(g):
				for kk in range(g):
					col = rgb2lab(np.array([min(ii*gsize,255),min(jj*gsize,255),min(kk*gsize,255)]))
					self.g_lab[ii][jj][kk] = col
					#col = rgb2lab(np.array([ii*gsize,jj*gsize,kk*gsize]))
					for i in range(self.K):
						for j in range(self.K):
							self.wi[ii][jj][kk][i] += lambda_ij[i,j] * phi(labDistance(col,self.orig_centers[j]))
					#clamp negative weight to zero and then renormalize
					#print("wi[{},{},{}]={}".format(ii,jj,kk,self.wi[ii][jj][kk]))
					renormal_flag = False
					for i in range(self.K):
						if self.wi[ii][jj][kk][i] < 0 :
							self.wi[ii][jj][kk][i] = 0
							renormal_flag = True
					
					sum_tmp = np.sum(self.wi[ii][jj][kk])
					#if renormal_flag:
					self.wi[ii][jj][kk] = self.wi[ii][jj][kk]/sum_tmp
					#else:
					#	print("sum_tmp:",sum_tmp)
					#print("wi[{},{},{}]={},flag={}".format(ii,jj,kk,self.wi[ii][jj][kk],renormal_flag))
	

		#precompute the weight between each pixel to its eight neareast grid values
		height,width = self.img.shape[0:2]
		
		
		orignal_img = self.orig_img/gsize
		p_w_idx0 = np.floor(orignal_img)

		def sumAbsMul(arr):
			arr = np.abs(arr)
			return arr[:,:,0]*arr[:,:,1]*arr[:,:,2]
		self.pixel_w = np.zeros((height,width,8))
		self.pixel_w[:,:,7] = sumAbsMul(p_w_idx0 - orignal_img)
		self.pixel_w[:,:,6] = sumAbsMul(p_w_idx0 + np.array([0,0,1]) - orignal_img)
		self.pixel_w[:,:,5] = sumAbsMul(p_w_idx0 + np.array([0,1,0]) - orignal_img)
		self.pixel_w[:,:,4] = sumAbsMul(p_w_idx0 + np.array([0,1,1]) - orignal_img)
		self.pixel_w[:,:,3] = sumAbsMul(p_w_idx0 + np.array([1,0,0]) - orignal_img)
		self.pixel_w[:,:,2] = sumAbsMul(p_w_idx0 + np.array([1,0,1]) - orignal_img)
		self.pixel_w[:,:,1] = sumAbsMul(p_w_idx0 + np.array([1,1,0]) - orignal_img)
		self.pixel_w[:,:,0] = sumAbsMul(p_w_idx0 + np.array([1,1,1]) - orignal_img)
		
		#time3 = time.time()
		


		#time5 = time.time()
		#print("{}s\n{}s\n{}s\n".format(time6-time1,time2-time6,time2-time1))
		#self.showCenters(ref_centers)

	def showCenters(self):
		self.showCenters(self.centers_lab)

	def showCenters(self,centers):
		blocksize = 80
		centers_rgb = lab2rgb4arr(centers)
		ncluster = centers.shape[0]
		img = np.zeros((blocksize,blocksize*ncluster,3))
		
		for i in range(ncluster):
			block = np.zeros((blocksize,blocksize,3))
			block[:,:] = centers_rgb[i]
			img[:,i*blocksize:(i+1)*blocksize,:] = block

		plt.imshow(np.array(img,dtype=np.uint8))
		plt.show()
	#return the rgb centers/palette
	def centers(self):
		res = lab2rgb4arr(self.centers_lab)
		res = np.where(res<0,0,res)
		res = np.where(res>255,255,res)
		return np.array(res,dtype=np.uint8)

	def norm2orig(self,val,axis='r'):
		if axis=='r':
			return int(round(val*self.bin_range[0]+self.bin_base[0]))
		if axis=='g':
			return int(round(val*self.bin_range[1]+self.bin_base[1]))
		if axis=='b':
			return int(round(val*self.bin_range[2]+self.bin_base[2]))
		return val
		
	def normalize(self,img):
		r = img[:,:,0].reshape(-1,1).copy()
		g = img[:,:,1].reshape(-1,1).copy()
		b = img[:,:,2].reshape(-1,1).copy()
		
		rmax,rmin = r.max(),r.min()
		gmax,gmin = g.max(),g.min()
		bmax,bmin = b.max(),b.min()
		r_norm = (r - rmin)/(rmax-rmin)
		g_norm = (g - gmin)/(gmax-gmin)
		b_norm = (b - bmin)/(bmax-bmin)
		img_data = np.concatenate((r_norm,g_norm,b_norm),axis=1)
		return (img_data,[r_norm,g_norm,b_norm],[rmin,gmin,bmin],[(rmax-rmin),(gmax-gmin),(bmax-bmin)])

		
	def kmeans(self,lock_black=True):
		centers = self.init_center4kmeans(lock_black)

		'''centers = np.array([[0,0,0],[41.25595568978259,-13.430644963549721,18.325524758002672], \
		[99.86456269387023,-0.3109940638383857,-0.08613797071691096], \
		[43.591213116910644,8.309873936857255,-30.38691603072777], \
		[57.229503813615054,30.26064995100793,31.614862560671707], \
		[45.24416783631266,-22.90240052940612,30.26613777060414]])'''

		#self.showCenters(centers)
		no_change = False

		valid_weight = np.squeeze((self.bin_count.reshape(-1,1) != 0),axis=1)
		bin_weight = self.bin_count.reshape(-1,1)[valid_weight]
		print("bin_weight=",bin_weight.shape)
		bin_color = self.bin_color.reshape(-1,3)[valid_weight]
		bin_tag = self.bin_tag.reshape(-1,1)[valid_weight]
		ncluster = centers.shape[0]
		iter_times = 0
		iter_MAXtimes = 20
		while (not no_change) and iter_times<iter_MAXtimes:
			#print("Iter:{},Kmeans centers:".format(iter_times))
			#for i in range(centers.shape[0]):
			#	print(centers[i])

			iter_times += 1
			#no_change = True
			summ = np.zeros((ncluster,3))
			count = np.zeros((ncluster,1))
			#pdb.set_trace()
			for i in range(bin_color.shape[0]):
				#if bin_weight[i] == 0:
				#	continue
				mind = self.inf
				mini = -1;
				#update the tag of each bin
				for p in range(ncluster):
					d2 = distance2(bin_color[i],centers[p])
					if d2<mind :
						mind = d2
						mini = p
				if mini != bin_tag[i]:
					bin_tag[i] = mini
					#no_change = False
				summ[mini] += bin_color[i]*bin_weight[i]
				count[mini] += bin_weight[i]
			#update cluster
			tmp_centers = centers.copy()
			sum_d2 = 0
			for i in range(ncluster):
				if lock_black==True and i==0:
					continue
				#tmp_centers[i] = centers[i]
				if count[i]!=0 :
					tmp_centers[i] = summ[i]/count[i]
				sum_d2 += distance(tmp_centers[i],centers[i])
			
			if math.sqrt(sum_d2)<0.01:
				no_change = True
			centers = tmp_centers
			
			#self.showCenters(centers)

		#discard the black palette
		if lock_black==True:
			centers = centers[1:]
		centers = self.sortbylumination(centers)
		assert (centers.shape[0] == self.K),"centers numbers are not equal to K"
		return centers
	
	def sortbylumination(self,arr):
		#in increasing order
		lumi = arr[:,0]
		sort_idx = np.argsort(lumi)
		copy = arr.copy()
		for i in range(sort_idx.size):
			copy[i,:] = arr[sort_idx[i],:]
		return copy

	def init_center4kmeans(self,lock_black=True):
		bin_black = rgb2lab(np.array([0,0,0]))
		#bin_black = rgb2lab(np.array([0,0,0]))
		centers_lab = np.zeros((self.K,3))
		valid_weight = np.squeeze(self.bin_count.reshape(-1,1),axis=1)
		#print("bin_weight.shape=",bin_weight2.shape)
		bin_weight = self.bin_count.reshape(-1,1)[valid_weight!=0].copy()
		#print("bin_weight2!=0",bin_weight2!=0)

		bin_copy = self.bin_color.reshape(-1,3)[valid_weight!=0]
		#print("bin_weight2!=0.shape=",(bin_weight2!=0).shape)
	

		#print("bin_weight.shape=",bin_weight.shape)
		#print("bin_copy.shape=",bin_copy.shape)
		#pdb.set_trace()

		#for i in range(bin_weight.shape[0]):
		#	d2 = distance2(bin_copy[i],bin_black)
		#	factor = 1 - math.exp(-d2*100)
		#	bin_weight[i] *= factor
		for p in range(self.K):		
			#tmp = None
			#maxc = -1		
			#for i in range(bin_weight.shape[0]):

				#if not p==0:
				#	d2 = distance2(bin_copy[i] , centers_lab[p-1])
				#	factor = 1 - math.exp(-d2/6400)
				#	bin_weight[i] *= factor
				#elif p==0 and lock_black:
					#d2 = distance2(bin_copy[i] , bin_black)
					#factor = 1 - math.exp(-(d2/8100))
				#	factor = 1
				#	bin_weight[i] *= factor
				#if (bin_weight[i]>maxc):
					#maxc = bin_weight[i]
					#tmp = bin_copy[i]
			for i in range(bin_weight.shape[0]):
				rgb = lab2rgb(bin_copy[i])
				#print("bin_color(({},{},{})).weight = {}".format(rgb[0],rgb[1],rgb[2],bin_weight[i]))
			idx = np.argmax(bin_weight)
			tmp = bin_copy[idx]
			#pdb.set_trace()
			centers_lab[p] = tmp
			for i in range(bin_weight.shape[0]):
				d2 = distance2(bin_copy[i] , tmp)
				factor = 1 - math.exp(-d2/6400)
				bin_weight[i] *= factor


			#self.showCenters(centers_lab[0:p+1])
		#pdb.set_trace()
		if lock_black==True:
			centers_lab = np.insert(centers_lab,0,bin_black,axis=0)

		#self.showCenters(centers_lab[0:self.K+1])
		return centers_lab


	def adjustCentersL(self,idx,col_lab):
		for i in range(idx+1,self.K,1):
			self.centers_lab[i][0] = max(self.user_edited[i][0],self.centers_lab[i-1][0])

		for i in range(idx-1,-1,-1):
			self.centers_lab[i][0] = min(self.user_edited[i][0],self.centers_lab[i+1][0])
		return 
 
	def getBoundary(self,src,dst):		
		assert (isLegalLab(src) and (not isLegalLab(dst))),'wrong src or dst lab-color while computing Lab-Boundary'
		#assert (src[0]==dst[0]),"src and dst are not in the same L-slice "
		src = src.copy()
		dst = dst.copy()
		while labDistance2(src,dst)>0.00001 :
			mid = (src+dst)/2
			if isLegalLab(mid):
				src = mid
			else:
				dst = mid
		return src
	def updateSingle_ab(self,col_lab,offset):

		return col_lab
	def updateGrid(self):
		g = self.g
		gsize = self.gsize
		g_lab = self.g_lab.reshape(-1,3)
		g_rgb = self.g_rgb.reshape(-1,3)
		wi = self.wi.reshape(-1,self.K)
		for ii in range(g_lab.shape[0]):		
			col_lab = g_lab[ii]
			#col_lab = rgb2lab(np.array([ii*gsize,jj*gsize,kk*gsize]))
					
			#update a,b channel
			dsti = np.repeat(col_lab[np.newaxis,:],self.K,axis=0)
			for i in range(self.K):
				if wi[ii][i]==0:
					continue
				if not np.any(self.user_edited[i]-self.orig_centers[i]):
					continue
				x = col_lab
						
				C = self.orig_centers[i].copy()
				Cprime = self.centers_lab[i].copy()
				#shift to the same L-slice
				C[0] = x[0]
				Cprime[0] = x[0]
				offset = Cprime - C
				x0 = x + offset
				Cb = labIntersect(C,Cprime)

				if isLegalLab(x0):
					xb = labIntersect(x,x0)
				else:
					xb = labBoundary(Cprime,x0)
				x2xb = labDistance(xb,x)
				xlen = min(1,x2xb/labDistance(Cb,C))
				xprime = x + (xb-x)/x2xb * xlen * labDistance(C,Cprime)
						
				dsti[i] = xprime
						
			dst = np.sum(dsti * wi[ii].reshape(-1,1),axis=0)

			newL = np.append(np.insert(self.centers_lab[:,0],0,0),100)
			for i in range(newL.shape[0]-1):
				if col_lab[0]>= self.origL[i] and col_lab[0]<=self.origL[i+1]:
					dst[0] = ((col_lab[0]-self.origL[i])*newL[i+1] + (self.origL[i+1]-col_lab[0])*newL[i])/(self.origL[i+1]-self.origL[i])
					break
					
			rgb = lab2rgb(dst)
			rgb = np.where(rgb>255,255,rgb) 
			g_rgb[ii] = np.where(rgb<0,0,rgb)
					
		
		self.g_rgb = g_rgb.reshape((g,g,g,3))

	
	def update(self,idx,col_rgb):
		start = time.time()
		#print("begin to update {} with color({},{},{})".format(idx,col_rgb[0],col_rgb[1],col_rgb[2]))
		col_lab = rgb2lab(col_rgb)

		#print("#before centers_lab:{} \n centers_rgb:{}".format(self.centers_lab,lab2rgb4arr(self.centers_lab)))

		#resimg = self.img.copy()

		#update centers
		self.centers_lab[idx] = col_lab
		self.user_edited[idx] = col_lab
		#adjust centers' luminance(l) to keep it monotonic
		self.adjustCentersL(idx,col_lab)
		#print("update_fun_centers_lab:{}\ncenters_rgb:{}".format(self.centers_lab,self.centers()))

		#update ab of each grid 
		height,width = self.img.shape[0:2]


		g = self.g
		gsize = int(math.ceil(256/(g-1)))
		
		#print("gsize:",gsize)
		begin_time = time.time()
		#update Grid
		#print("updating Grid...")
		#t1 = time.time()
		self.updateGrid()
		#t2 = time.time()
		g = self.g
		gsize = self.gsize

		idx = np.floor(self.orig_img/gsize)
		idx = idx[:,:,np.newaxis,:]
		
		rgb_grid = np.repeat(idx,8,axis=2)
		rgb_grid[:,:,1] += np.array([0,0,1])
		rgb_grid[:,:,2] += np.array([0,1,0])
		rgb_grid[:,:,3] += np.array([0,1,1])
		rgb_grid[:,:,4] += np.array([1,0,0])
		rgb_grid[:,:,5] += np.array([1,0,1])
		rgb_grid[:,:,6] += np.array([1,1,0])
		rgb_grid[:,:,7] += np.array([1,1,1])
		#rgb_grid = rgb_grid * gsize
		rgb_idx = np.array(rgb_grid.reshape(-1,3),dtype=np.int_)
		#print("rgb_idx:",rgb_idx)
		r_idx = rgb_idx[:,0]
		g_idx = rgb_idx[:,1]
		b_idx = rgb_idx[:,2]
		new_grid = self.g_rgb[r_idx,g_idx,b_idx].reshape(height,width,8,3)
		
		#print(img.shape)
		img = new_grid#rgb_grid*gsize
		weight = self.pixel_w.reshape((height,width,8,1))
		weight = np.repeat(weight,3,axis=3)
		#Triliner interpolation
		img = img * weight
		#t3 = time.time()
		img = np.sum(img,axis=2)
		img = np.around(img)

		#end = time.time()
		#print("Totol time:{}+{}+{}+{}={}".format(t1-start,t2-t1,t3-t2,end-t3,end-start))
		return np.array(img,dtype=np.uint8)