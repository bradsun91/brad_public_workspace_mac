# -*- coding: utf-8 -*- 
'''
TopQuant-简称TQ极宽智能量化回溯分析系统，培训课件-配套教学python程序

Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2017.10.1 首发


网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
  
文件名:ztools_mxnet.py
默认缩写：import ztools_mxnet as zmx
简介：Top极宽量化·AI智能模块·MXNET工具函数库
 

'''
#

import sys,os,re
import arrow,bs4,random
import numexpr as ne  
import numpy as np
import pandas as pd
import tushare as ts
#import talib as ta

import pypinyin 
#

import matplotlib as mpl
from matplotlib import pyplot as plt

#
#
from mxnet import gluon, autograd, ndarray as nd
from mxnet.gluon import nn
import mxnet as mx

#
import zsys
import zpd_talib as zta
#
import ztools as zt
import ztools_tq as ztq
import ztools_str as zstr
import ztools_sta as zsta
import ztools_data as zdat
import ztools_draw as zdr

 
#-------------------

#-------------------dat,xxx
def df_xcov2nd(df,xlst=['avg'],ylst=['avg2'],d1k=1000,fgGPU=False):
    '''
    把输入的数据df，从pdans的DataFrame表格格式，转换为MXNet的NDAarray矩阵格式，方便MXNEt神经网络模型使用。
输入参数：
	df，输入数据变量
	xlst，模型输入参数字段名称。
	ylst，数据标签字段名称。
	d1k，部分数据需要进行大小调整，默认为缩写1000倍。
	fgGPU，GPU设备开关标志，默认使用cpu作为计算设备。
返回参数：
	x_dat，模型计算数据集，MXNet的NDAarray矩阵格式。
	y_dat，模型标签数据集，MXNet的NDAarray矩阵格式。

    '''
    #
    vlst=list(set(xlst)|set(ylst))
    for xsgn in vlst:
        df[xsgn]=df[xsgn]/d1k
    #--------------
    x_dat0,y_dat0=df[xlst].values,df[ylst].values
    #
    if fgGPU:xdev=mx.gpu()
    else:xdev=mx.cpu()
    #    
    x_dat,y_dat= nd.array(x_dat0,ctx=xdev),nd.array(y_dat0,ctx=xdev)
    #
    return x_dat,y_dat    

#-------------------mod,xxx
    
 
def mod_pred(model,xtst,df_tst,ftg='tmp/dpred100.csv'):
    '''
    使用训练好的模型，根据输入数据xtst，生成预测数据集。
输入参数：
	model，训练好的神经网络模型。
	xtst，输入数据集，格式要符合model模型的要求。
	df_tst，包含xtst的数据集，使用pandas的DataFrame表格格式。
	ftg，结果数据保存文件名，默认为：tmp/dpred100.csv。
返回参数：
	df_tst，包含预测数据的数据集，使用pandas的DataFrame表格格式。
'''
    y_pred0 = model(xtst)
    y_pred=y_pred0.asnumpy()
    df_tst['y_pred']=y_pred
    #df_tst['y_pred']=df_tst['y_pred0']
    print('\ndf_tst')
    print(df_tst.tail(10))
    #
    if ftg!='':
        df_tst.to_csv(ftg)
    #
    return df_tst    
    
def mod_outsym(model,fn0='tmp/mx001'):
    '''
    根据训练好模型model，生成图形格式的神经网络模型，需要预先安装好graphviz程序和pydot模块库。
输入参数：
	model，训练好的神经网络模型。
	fn0，模型输出文件前缀，默认为：'tmp/mx001。
返回参数：
	无。
运行后，会在tmp目录下生成一个json文件，并在程序目录下面生成一个pdf文件，pdf文件当中，有绘制好的神经网络模型结构图。
    '''

    model.export(fn0)
    #
    fmod=fn0+'-symbol.json'
    symnet = mx.symbol.load(fmod)
    mx.viz.plot_network(symnet).view()    
    #
    print('\n@f,',fmod)
    
    