# -*- coding:utf-8 -*-
 
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import glob
 
#包含的汉字列表（太长，仅仅截取了一部分）
hanzi = u'0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz的一是不人有了在你我个大中要这为上生时会以就子到来可能和自们年多发心好用家出关长他成天对也小后下学都点国过地行信方得最说二业分作如看女于面注别经动公开现而美么还事'
 
#生成文字矩阵
def gen_img(text, size=(48,48), fontname='simhei.ttf', fontsize=48):
	im = Image.new('1', size, 1)
	dr = ImageDraw.Draw(im)
	font = ImageFont.truetype(fontname, fontsize)
	dr.text((0, 0), text, font=font)
	return (((np.array(im.getdata()).reshape(size)==0)+(np.random.random(size)<0.05)) != 0).astype(float)
 
#生成训练样本
data = pd.DataFrame()
fonts = glob.glob('./*.[tT][tT]*')
for fontname in fonts:
	print fontname
	for i in range(-2,3):
		m = pd.DataFrame(pd.Series(list(hanzi)).apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48+i)]))
		m['label'] = range(3062)
		data = data.append(m, ignore_index=True)
		m = pd.DataFrame(pd.Series(list(hanzi)).apply(lambda s:[gen_img(s, fontname=fontname, fontsize=48+i)]))
		m['label'] = range(3062)
		data = data.append(m, ignore_index=True)
 
 
x = np.array(list(data[0])).astype(float)
np.save('x', x) #保存训练数据
 
dic=dict(zip(range(3062),list(hanzi))) #构建字表
 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
 
batch_size = 1024
nb_classes = 3062
nb_epoch = 30
 
img_rows, img_cols = 48, 48
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4
 
x = np.load('x.npy')
y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
weight = ((3062-np.arange(3062))/3062.0+1)**3
weight = dict(zip(range(3063),weight/weight.mean())) #调整权重，高频字优先
 
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
 
history = model.fit(x, y,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    class_weight=weight)
 
score = model.evaluate(x,y)
print('Test score:', score[0])
print('Test accuracy:', score[1])
 
model.save_weights('model.model')
