# -*- coding:utf-8 -*-
 
import numpy as np
from scipy import misc,ndimage
from scipy.stats import gaussian_kde as kde
from tqdm import *
 
def myread(filename): #读取图像，放大两倍，做平方变换
	print u'读取图片中...'
	pic = misc.imread(filename, flatten = True)
	pic = ndimage.zoom(pic, 2)
	pic = pic**2
	pic = ((pic-pic.min())/(pic.max()-pic.min())*255).round()
	print u'读取完成.'
	return pic
 
 
def decompose(pic): #核密度聚类，给出极大值、极小值点、背景颜色、聚类图层
	print u'图层聚类分解中...'
	d0 = kde(pic.reshape(-1), bw_method=0.2)(range(256)) #核密度估计
	d = np.diff(d0)
	d1 = np.where((d[:-1]<0)*(d[1:]>0))[0] #极小值
	d1 = [0]+list(d1)+[256]
	d2 = np.where((d[:-1]>0)*(d[1:]<0))[0] #极大值
	if d1[1] < d2[0]:
		d2 = [0]+list(d2)
	if d1[len(d1)-2] > d2[len(d2)-1]:
		d2 = list(d2)+[255]
	dc = sum(map(lambda i: d2[i]*(pic >= d1[i])*(pic < d1[i+1]), range(len(d2))))
	print u'分解完成. 共%s个图层'%len(d2)
	return dc
 
 
def erosion_test(dc): #抗腐蚀能力测试
	print u'抗腐蚀能力测试中...'
	layers = []
	#bg = np.argmax(np.bincount(dc.reshape(-1)))
	#d = [i for i in np.unique(dc) if i != bg]
	d = np.unique(dc)
	for k in d:
		f = dc==k
		label_im, nb_labels = ndimage.label(f, structure=np.ones((3,3))) #划分连通区域
		ff = ndimage.binary_erosion(f) #腐蚀操作
		def test_one(i):
			index = label_im==i
			if (1.0*ff[index].sum()/f[index].sum() > 0.9) or (1.0*ff[index].sum()/f[index].sum() < 0.1):
				f[index] = False
		ff = map(test_one, trange(1, nb_labels+1))
		layers.append(f)
	print u'抗腐蚀能力检测完毕.'
	return layers
 
 
def pooling(layers): #以模仿池化的形式整合特征
	print u'整合分解的特征中...'
	result = sum(layers)
	label_im, nb_labels = ndimage.label(result, structure=np.ones((3,3)))
	def pool_one(i):
		index = label_im==i
		k = np.argmax([1.0*layers[j][index].sum()/result[index].sum() for j in range(len(layers))])
		result[index] = layers[k][index]
	t = map(pool_one, trange(1, nb_labels+1))
	print u'特征整合成功.'
	return result
 
 
def post_do(pic):
	label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
	print u'图像的后期去噪中...'
	def post_do_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		ss = 1.0 * len(pic.reshape(-1))/len(pic[index2].reshape(-1))**2
		#先判断是否低/高密度区，然后再判断是否孤立区。
		if (index.sum()*ss < 16) or ((1+len(pic[index2].reshape(-1))-index.sum())*ss < 16):
			pic[index] = False
		else:
			a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
			index3 = (slice(max(0, 2*a-b),min(pic.shape[0], 2*b-a)), slice(max(0, 2*c-d),min(pic.shape[1], 2*d-c)))
			if (pic[index3].sum() == index.sum()) and (1.0*index.sum()/(b-a)/(d-c) > 0.75):
				pic[index2] = False	
	t = map(post_do_one, trange(1, nb_labels+1))
	print u'后期去噪完成.'
	return pic
 
 
def areas(pic): #圈出候选区域
	print u'正在生成候选区域...'
	pic_ = pic.copy()
	label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
	def areas_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		pic_[index2] = True
	t = map(areas_one, trange(1, nb_labels+1))
	return pic_
 
 
#定义距离函数，返回值是距离和方向
#注意distance(o1, o2)与distance(o2, o1)的结果是不一致的
def distance(o1, o2): 
	delta = np.array(o2[0])-np.array(o1[0])
	d = np.abs(delta)-np.array([(o1[1]+o2[1])/2.0, (o1[2]+o2[2])/2.0])
	d = np.sum(((d >= 0)*d)**2)
	theta = np.angle(delta[0]+delta[1]*1j)
	k = 1
	if np.abs(theta) <= np.pi/4:
		k = 4
	elif np.abs(theta) >= np.pi*3/4:
		k = 2
	elif np.pi/4 < theta < np.pi*3/4:
		k = 1
	else:
		k = 3
	return d, k
 
 
def integrate(pic, k=0): #k=0是全向膨胀，k=1仅仅水平膨胀
	label_im, nb_labels = ndimage.label(pic, structure=np.ones((3,3)))
	def integrate_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
		cc = ((a+b)/2.0,(c+d)/2.0)
		return (cc, b-a, d-c)
	print u'正在确定区域属性...'
	A = map(integrate_one, trange(1, nb_labels+1))
	print u'区域属性已经确定，正在整合邻近区域...'
	aa,bb = pic.shape
	pic_ = pic.copy()
	def areas_one(i):
		dist = [distance(A[i-1], A[j-1]) for j in range(1, nb_labels+1) if i != j]
		dist = np.array(dist)
		ext = dist[np.argsort(dist[:,0])[0]] #通过排序找最小，得到最邻近区域
		if ext[0] <= (min(A[i-1][1],A[i-1][2])/4)**2:
			ext = int(ext[1])
			index = label_im==i
			index2 = ndimage.find_objects(index)[0]
			a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
			if ext == 1: #根据方向来膨胀
				pic_[a:b, c:min(d+(d-c)/4,bb)] = True
			elif ext == 3:
				pic_[a:b, max(c-(d-c)/4,0):d] = True
			elif ext == 4 and k == 0:
				pic_[a:min(b+(b-a)/6,aa), c:d] = True #基于横向排版假设，横向膨胀要大于竖向膨胀
			elif k == 0:
				pic_[max(a-(b-a)/6,0):b, c:d] = True
	t = map(areas_one, trange(1, nb_labels+1))
	print u'整合完成.'
	return pic_
 
 
def cut_blank(pic): #切除图片周围的白边，返回范围
	try:
		q = pic.sum(axis=1)
		ii,jj = np.where(q!= 0)[0][[0,-1]]
		xi = (ii, jj+1)
		q = pic.sum(axis=0)
		ii,jj = np.where(q!= 0)[0][[0,-1]]
		yi = (ii, jj+1)
		return [xi, yi]
	except:
		return [(0,1),(0,1)]
 
 
def trim(pic, pic_, prange=5): #剪除白边，删除太小的区域
	label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
	def trim_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		box = (pic*index)[index2]
		[(a1,b1), (c1,d1)] = cut_blank(box)
		pic_[index] = False
		if (b1-a1 < prange) or (d1-c1 < prange) or ((b1-a1)*(d1-c1) < prange**2): #删除小区域
			pass
		else: #恢复剪除白边后的区域
			a,b,c,d = index2[0].start, index2[0].stop, index2[1].start, index2[1].stop
			pic_[a+a1:a+b1,c+c1:c+d1] = True
	t = map(trim_one, trange(1, nb_labels+1))
	return pic_
 
 
def bound(m):
	frange = (slice(m.shape[0]-1), slice(m.shape[1]-1))
	f0 = np.abs(np.diff(m, axis=0))
	f1 = np.abs(np.diff(m, axis=1))
	f2 = np.abs(m[frange]-m[1:,1:])
	f3 = f0[frange]+f1[frange]+f2[frange] != 0
	return f3
 
 
def trim_bound(pic, pic_): #剪除白边，删除太小的区域
	pic_ = pic_.copy()
	label_im, nb_labels = ndimage.label(pic_, structure=np.ones((3,3)))
	def trim_one(i):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		box = pic[index2]
		if 1.0 * bound(box).sum()/box.sum() < 0.15:
			pic_[index] = False
	t = map(trim_one, trange(1, nb_labels+1))
	return pic_
