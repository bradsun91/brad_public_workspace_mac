# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 

网站： www.TopQuant.vip      www.ziwang.com
QQ群: Top极宽量化1群，124134140
      Top极宽量化2群，650924099
      Top极宽量化3群，450853713
  
    
TopQuant.vip ToolBox 2016
Top极宽·量化开源工具箱 系列软件 
by Top极宽·量化开源团队 2016.12.25 首发
  
文件名:ztools_draw.py
默认缩写：zdr,示例：import ztools_draw as zdr
简介：Top极宽量化软件，matplotlib绘图模块
'''


import sys,os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
#from PIL import Image,ImageDraw,ImageFont
#
#


import plotly as py
import plotly.graph_objs as pygo
from plotly import tools
from plotly.graph_objs import *
from plotly.graph_objs import Scatter, Layout, Figure
#from plotly.tools import FigureFactory as pyff
import plotly.figure_factory  as pyff
#
from sklearn import metrics
#
import zsys
import ztools as zt
import ztools_data as zdat

'''
var&const
#

misc
#
dr.mul.xxx
dr.fintech
#

'''
#----var&const.pre_def
pyplt=py.offline.plot

#----------dr.xxx
#
     
#----------drm.sub.xxx   
def drm_lay(df,m_title,m_dtick=10,m_tkAng=-20):
    lay = pygo.Layout( 
        title=m_title,
        xaxis=pygo.XAxis(
            gridcolor='rgb(180, 180, 180)',
            mirror='all',
            showgrid=True,
            showline=True,
            ticks='outside',#'inside',
            #
            dtick=m_dtick,
            tickangle=m_tkAng,
            #
            type='category',
            categoryarray=df.index,
        ),
    )
    #
    return lay


def drm_cdlsub(df,hdr='K线图'):
    '''
    根据输入数据df，绘制k线图。
输入参数：
		df，输入数据集变量，数据数据必须包括标准的OHLC数据和volume成交量数据字段。
	hdr，图片标题，默认为“k线图”。
	ftg，输出文件名，默认为：mp/tmp_plotly.html。
返回参数：
	无

    '''
    fig=pyff.create_candlestick(df.open, df.high, df.low, df.close, dates=df.index)
    fig['layout'].update(title=hdr,
        xaxis=pygo.XAxis(
            autorange=True,
            gridcolor='rgb(180, 180, 180)',
            mirror='all',
            showgrid=True,
            showline=True,
            ticks='outside',
            tickangle=-20,
            dtick=10,
            type='category',
            categoryarray=df.index,
            ),
        yaxis=pygo.YAxis(
            autorange=True,
            gridcolor='rgb(180, 180, 180)',
            ),
        yaxis2=pygo.YAxis(
                side='right', 
                overlaying='y',
                range=[0,max(df['volume'])*3],
            ),
    )  # fig.update
    r_vol = pygo.Bar(
            x=df.index,#df['xtim'],
            y=df['volume'],
            name='volume',
            yaxis= 'y2',
            opacity=0.6,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(color='rgb(8,48,107)',width=1.5,),
            ),
    )   #r_vol
    #
    fig['data'].extend([r_vol])
            
    #
    return fig   
         
def drm_ptsub(df,vsgn,wid=1,cor='',syb='circle'):
    # syb:xxx-open , circle square diamond cross x triangle star octagon hexagram  hexagram2
    rx = pygo.Scatter(
            x=df.index,#df['xtim'],
            y=df[vsgn],
            #name=vsgn,
            name='',
            showlegend=False,
            #connectgaps=True,
            mode = 'markers',
            #if v<0:cor=''.join(['rgb(',str(abs(v)),',0,0)'])
            #if v>0:cor=''.join(['rgb(0,0,',v,')'])
            marker=dict(color=cor, size=wid,symbol=syb)
        )
    #
    return rx   

def drm_lnsub(df,vsgn,wid=2,cor=''):
    rx = pygo.Scatter(
            x=df.index,#df['xtim'],
            y=df[vsgn],
            name=vsgn,
            mode = 'lines',
            #mode = 'lines+markers',
            #
            line=dict(width=wid,color=cor),
    )
    #
    return rx
            

#----------------drm.xxxx
def drm_pt(df,hdr='多维数据图',xlst=['avg','high','low'],wid=2,ftg='tmp/tmp_plotly.html'):
    '''
    根据输入数据df，绘制多散点图。
输入参数：
	df，输入数据集变量，数据数据必须包括xlst列表当中的数据字段。
	xlst，需要绘制的数据字段名称。
	wid，点图直径宽度，默认为2。
	ftg，输出文件名，默认为：mp/tmp_plotly.html。
返回参数：
	无

    '''
    xdat = pygo.Data()
    #
    vlst=['circle','square','diamond','x','star','octagon','hexagram','triangle','cross']
    vc=0
    # syb:xxx-open , circle square diamond cross x triangle star octagon hexagram  hexagram2
    for xsgn in xlst:
        rx=drm_ptsub(df,xsgn,wid,syb=vlst[vc])
        vc+=1
        xdat.extend([rx])
    #
    lay=drm_lay(df,hdr,m_dtick=20,m_tkAng=-20)
    fig = pygo.Figure(data=xdat, layout=lay)
    pyplt(fig,filename=ftg,show_link=False)
    
   
def drm_line(df,hdr='多维数据图',xlst=['avg','high','low'],p0wid=3,wid=1,ftg='tmp/tmp_plotly.html'):
    '''
    根据输入数据df，绘制多条曲线图。
输入参数：
	df，输入数据集变量，数据数据必须包括xlst列表当中的数据字段。
	xlst，需要绘制的数据字段名称。
	wid，曲线宽度，默认为2。
	ftg，输出文件名，默认为：mp/tmp_plotly.html。
返回参数：
	无
'''
    xdat = pygo.Data()
    #
    xc=0
    for xsgn in xlst:
        if xc==0:
            rx=drm_lnsub(df,xsgn,p0wid)
        else:
            rx=drm_lnsub(df,xsgn,wid)
        #    
        xc+=1
        xdat.extend([rx])
    #
    lay=drm_lay(df,hdr,m_dtick=20,m_tkAng=-20)
    fig = pygo.Figure(data=xdat, layout=lay)
    pyplt(fig,filename=ftg,show_link=False)
    
#-----------



def drm_cdl(df,hdr='K线图',ftg='tmp/tmp_plotly.html'):
    '''
    根据输入数据df，绘制k线图。
输入参数：
		df，输入数据集变量，数据数据必须包括标准的OHLC数据和volume成交量数据字段。
	hdr，图片标题，默认为“k线图”。
	ftg，输出文件名，默认为：mp/tmp_plotly.html。
返回参数：
	无

    '''
    fig=drm_cdlsub(df,hdr)
    #
    pyplt(fig,filename=ftg,show_link=False)   
    
def drm_xcdl(df,hdr='K线图',xlst=[],ftg='tmp/tmp_plotly.html'):
    '''
    根据输入数据df，绘制k线图。
输入参数：
		df，输入数据集变量，数据数据必须包括标准的OHLC数据和volume成交量数据字段。
	hdr，图片标题，默认为“k线图”。
	ftg，输出文件名，默认为：mp/tmp_plotly.html。
返回参数：
	无

    '''
    fig=drm_cdlsub(df,hdr)
    #
    for xsgn in xlst:
        rx=drm_lnsub(df,xsgn,1)
        #xdat.extend([rx])
        fig['data'].extend([rx])
    #
    pyplt(fig,filename=ftg,show_link=False)   
        
#---------drm.x.xxx    
    
def drm_xline(df,hdr='多维数据图',xlst=['t1','t5','t15'],p9df=None,p1df=None,psgn='avg',p9syb='star',p1syb='x',ftg='tmp/tmp_plotly.html'):
    xdat = pygo.Data()
    #
    for xsgn in xlst:
        rx=drm_lnsub(df,xsgn,2)
        xdat.extend([rx])
    #
    # -open , circle square diamond cross x triangle star octagon hexagram  hexagram2
    #
    if isinstance(p9df, pd.DataFrame):
        rx=drm_ptsub(p9df,psgn,10,cor='rgb(250,0,0)',syb=p9syb)
        xdat.extend([rx])
    #
    if isinstance(p1df, pd.DataFrame):
        rx=drm_ptsub(p1df,psgn,10,cor='rgb(0,0,250)',syb=p1syb)
        xdat.extend([rx])
    #
    
    lay=drm_lay(df,hdr,m_dtick=20,m_tkAng=-20)
    fig = pygo.Figure(data=xdat, layout=lay)
    #
    pyplt(fig,filename=ftg,show_link=False)
            
