# -*- coding:utf-8 -*-
 
import numpy as np
from scipy import misc
from images import cut_blank
 
#包含的汉字列表（太长了，仅截取了一部分）
hanzi = u'0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz的一是不人有了在你我个大中要这为上生时会以就子到来可能和自们年多发心好用家出关长他成天对也小后下学都点国过地行信方得最说二业分作如看女于面注别经动公开现而美么还事'
 
dic=dict(zip(range(3062),list(hanzi))) #构建字表
 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
 
batch_size = 128
nb_classes = 3062
img_rows, img_cols = 48, 48
nb_filters = 64
nb_pool = 2
nb_conv = 4
 
model = Sequential()
 
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
model.load_weights('ocr.model')
 
import pandas as pd
zy = pd.read_csv('zhuanyi.csv', encoding='utf-8', header=None)
zy.set_index(0, inplace=True)
zy = zy[1]
 
def viterbi(nodes):
	paths = nodes[0]
	for l in range(1,len(nodes)):
		paths_ = paths.copy()
		paths = {}
		for i in nodes[l].keys():
			nows = {}
			for j in paths_.keys():
				try:
					nows[j+i]= paths_[j]*nodes[l][i]*zy[j[-1]+i]
				except:
					nows[j+i]= paths_[j]*nodes[l][i]*zy[j[-1]+'XX']
			k = np.argmax(nows.values())
			paths[nows.keys()[k]] = nows.values()[k]
	return paths.keys()[np.argmax(paths.values())]
 
 
# mode为direact和search
#前者直接给出识别结果，后者给出3个字及其概率（用来动态规划）
def ocr_one(m, mode='direact'):
	m = m[[slice(*i) for i in cut_blank(m)]]
	if m.shape[0] >= m.shape[1]:
		p = np.zeros((m.shape[0],m.shape[0]))
		p[:,:m.shape[1]] = m
	else:
		p = np.zeros((m.shape[1],m.shape[1]))
		x = (m.shape[1]-m.shape[0])/2
		p[:m.shape[0],:] = m
	m = misc.imresize(p,(46,46), interp='nearest') #这步和接下来几步，归一化图像为48x48
	p = np.zeros((48, 48))
	p[1:47,1:47] = m 
	m = p
	m = 1.0 * m / m.max()
	k = model.predict(np.array([[m]]), verbose=0)[0]
	ks = k.argsort()
	if mode == 'direact':
		if k[ks[-1]] > 0.5:
			return dic[ks[-1]]
		else:
			return ''
	elif mode == 'search':
		return {dic[ks[-1]]:k[ks[-1]],dic[ks[-2]]:k[ks[-2]],dic[ks[-3]]:k[ks[-3]]}
 
'''
#直接调用Tesseract
import os
def ocr_one(m):
	misc.imsave('tmp.png', m)
	os.system('tesseract tmp.png tmp -l chi_sim -psm 10')
	s = open('tmp.txt').read()
	os.system('rm tmp.txt \n rm tmp.png')
	return s.strip()
'''
 
 
def cut_line(pl): #mode为direact或viterbi
	pl = pl[[slice(*i) for i in cut_blank(pl)]]
	pl0 = pl.sum(axis=0)
	pl0 = np.where(pl0==0)[0]
	if len(pl0) > 0:
		pl1=[pl0[0]]
		t=[pl0[0]]
		for i in pl0[1:]:
			if i-pl1[-1] == 1:
				t.append(i)
				pl1[-1]=i
			else:
				pl1[-1] = sum(t)/len(t)
				t = [i]
				pl1.append(i)
		pl1[-1] = sum(t)/len(t)
		pl1 = [0] + pl1 + [pl.shape[1]-1]
		cut_position = [1.0*(pl1[i+1]-pl1[i-1])/pl.shape[0] > 1.2 for i in range(1,len(pl1)-1)]
		cut_position=[pl1[1:-1][i] for i in range(len(pl1)-2) if cut_position[i]] #简单的切割算法
		cut_position = [0] + cut_position + [pl.shape[1]-1]
	else:
		cut_position = [0, pl.shape[1]-1]
	l = len(cut_position)
	for i in range(1, l):
		j = int(round(1.0*(cut_position[i]-cut_position[i-1])/pl.shape[0]))
		ab = (cut_position[i]-cut_position[i-1])/max(j,1)
		cut_position = cut_position + [k*ab+cut_position[i-1] for k in range(1, j)]
	cut_position.sort()
	return pl, cut_position
 
 
def ocr_line(pl, mode='viterbi'): #mode为direact或viterbi
	pl, cut_position = cut_line(pl)
	if mode == 'viterbi':
		text = map(lambda i: ocr_one(pl[:,cut_position[i]:cut_position[i+1]+1], mode='search'), range(len(cut_position)-1))
		return viterbi(text)
	elif mode == 'direact':
		text = map(lambda i: ocr_one(pl[:,cut_position[i]:cut_position[i+1]+1]), range(len(cut_position)-1))
		''.join(text)
