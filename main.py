# -*- coding:utf-8 -*-
 
from scipy import ndimage
print u'加载图片工具中...'
from images import *
print u'加载OCR模型中...'
from ocr import *
print u'加载完毕.'
 
if __name__ == '__main__':
	filename = '../cn.jpg'
	p = myread(filename)
	dc = decompose(p)
	layers = erosion_test(dc)
	result = pooling(layers)
	result = post_do(result)
	result_ = areas(result)
	result_ = integrate(result_, 1)
	result_ = trim(result, result_)
	result_ = integrate(result_, 1)
	result_ = trim(result, result_, 10)
	result_ = trim_bound(result, result_)
	label_im, nb_labels = ndimage.label(result_, structure=np.ones((3,3)))
	for i in range(1, nb_labels+1):
		index = label_im==i
		index2 = ndimage.find_objects(index)[0]
		print ocr_line(result[index2])
