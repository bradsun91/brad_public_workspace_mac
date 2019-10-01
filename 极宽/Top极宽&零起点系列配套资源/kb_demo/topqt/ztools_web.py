#coding=utf-8
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
  
文件名:ztools_web.py
默认缩写：import ztools_web as zweb
简介：Top极宽web网络与htm网页常用工具函数集
'''

import os,sys,io,re
import arrow,bs4,random
import pandas as pd
import tushare as ts
#
import requests
import bs4
from bs4 import BeautifulSoup 
from robobrowser import RoboBrowser 
from concurrent import futures
#from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
#

import zsys
import ztools as zt
import ztools_str as zstr
import ztools_data as zdat
#
#-----------------------
'''
xxx.var&const
misc
#
web_get_xxx    
web_get_xxx.site...
#
#--web_dz.xxx   ,discuz
#----zdz.zwx.xxx   zdz.zwx-->zdzx
#---bs4.xxx
'''
#-----------------------

#------------web.var&const
zt_headers = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"}
zt_xagent='Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'

                       

#-----------------------


    
#---web_get_xxx    
    
def web_get001sub(url):
    try:
        rx= requests.get(url,headers=zt_headers)  #获得网页,headers
    except:
        rx=None
        return None
    finally:
        return rx
    #
    return rx    
   
def web_get001(url):
    rx,xc=None,0
    while (rx is None)or(xc<3):
        rx=web_get001sub(url)
        xc+=1
    #
    return rx    

def web_get001txt(url,ucod='gb18030',ftg='',fcod='gbk'):
    htm,rx='',web_get001(url)
    if rx!=None:
        xcod=rx.apparent_encoding;#print(xcod,uss)
        rx.encoding =xcod  #gb-18030
        #dss=rx.text
        htm=rx.text;#print(htm)
        if xcod.upper()=='UTF-8':
            #print('@@u8a');#print(htm)
            htm=htm.replace('&nbsp;',' ')
            css = htm.encode("UTF-8",'ignore').decode("UTF-8",'ignore')
            css=css.replace(u'\xfffd ', u' ')
            css=css.replace(u'\xa0 ', u' ')
            htm = css.encode("GBK",'ignore').decode("GBK",'ignore')
        #
        if ftg!='':zt.f_add(ftg,htm,True,cod=fcod)
    #
    return htm
    

    
def web_get001txtFg(uss,fss):
    fsiz=zt.f_size(fss)
    if zsys.web_get001txtFg or(fsiz<1000):
        #print(zsys.sgnSP8,fss,fsiz)
        #uss=xtfb.us0_extOuzhi+xtfb.kgid+'.shtml';#print(uss)
        htm=web_get001txt(uss,ftg=fss,fcod='GB18030');#print(fss)
    else:
        htm=zt.f_rd(fss,cod='GB18030')    
    #
    return htm

def web_get001txtXFg(uss,fss,downFg=True):
    fsiz=zt.f_size(fss)
    if downFg or(fsiz<1000):
        #print(zsys.sgnSP8,fss,fsiz)
        #uss=xtfb.us0_extOuzhi+xtfb.kgid+'.shtml';#print(uss)
        htm=web_get001txt(uss,ftg=fss,fcod='GB18030');#print(fss)
    else:
        htm=zt.f_rd(fss,cod='GB18030')    
    #
    return htm
    
    
def web_getXLnks(url,ckn=10,kget=None,kflt=None,uget=None,uflt=None,ucod='gbk'):
    #rx= requests.get(url,headers=zt_headers)  #获得网页,headers
    #print(url)
    df=pd.DataFrame(columns=['hdr','url'])
    rx=web_get001(url)
    if rx==None:return df
    #
    #rx.encoding =ucod #gb-18030
    bs=BeautifulSoup(rx.text,'html5lib') # 'lxml'
    #bs=bs0.prettify('utf-8')
    xlnks=bs.find_all('a');#print(xlnks)
    
    ds=pd.Series(['',''],index=['hdr','url'])
    #print('\ncss,xss:',klnk,kflt)
    for lnk in xlnks:
        css,uss=lnk.text,lnk.get('href')
        #print('cs0,',css,uss)
        #
        if uflt!=None and uss!=None and zstr.str_xor(uss,uflt):uss=None
        if uget!=None and uss!=None and (not zstr.str_xor(uss,uget)):uss=None
        #
        if kflt!=None and uss!=None and zstr.str_xor(css,kflt):uss=None
        if kget!=None and uss!=None and (not zstr.str_xor(css,kget)):uss=None
        #print('cs2,',css,uss)
        #
        if uss==None:css=''
        css=zstr.str_fltHtmHdr(css)
        if len(css)>ckn:
            css=css.replace(',','，')
            #print('css,xss:',css,uss)
            ds['hdr'],ds['url']=css,uss
            df=df.append(ds.T,ignore_index=True)
    #
    #print(df)    
    return df    
    


#--- web_get_xxx.site...
            
def web_get_bdnews010(kstr,pn=1):
    url_bdnews0='http://news.baidu.com/ns?cl=2&ct=0&rn=50&ie=gbk&word={0}&pn={1}'  #pn=50x
    #
    df9=pd.DataFrame(columns=['hdr','url'])
    for xc in range(0,pn):
        uss=url_bdnews0.format(kstr,xc*50);print(uss)
        df=web_getXLnks(uss);
        df9=df9.append(df,ignore_index=True)
    #
    df9=df9.drop_duplicates(['hdr'])
    return df9
    
def web_get_cnblog010(kstr,timSgn='OneWeek',npg=2):
    us0='http://zzk.cnblogs.com/s/blogpost?DateTimeRange='+timSgn+'&Keywords={0}&pageindex={1}'
    df9=pd.DataFrame(columns=['hdr','url'])
    for xc in range(0,npg):
        uss=us0.format(kstr,xc);print(uss)
        df=web_getXLnks(uss);
        df9=df9.append(df,ignore_index=True)
    #
    df9=df9.drop_duplicates(['hdr'])
    return df9
    
   
def web_get_zhihu010(kstr):
    #1d=day;1w=week
    uss='https://www.zhihu.com/search?type=content&range=1w&q={0}'.format(kstr);
    #print( uss)
    df=web_getXLnks(uss,uget=['/question'],uflt=['/answer']);#print(df)
    #https://www.zhihu.com/question/21063634
    if len(df['hdr'])>0:
        df['url']='https://www.zhihu.com'+df['url']
        print(df)
        
    
    return df
#-----------------------

#---------------------zmul.web.xxx
        
def zmul_finx2urls(pn9=9,finx='dat/zw_bbs30k.csv',us0='http://ziwang.com/'):
    df_inx=pd.read_csv(finx,index_col=False,encoding='gbk')    #print(df_inx.tail())
    urls=[]
    for i, row in df_inx.iterrows():
        fid=row['uid']
        us2=''.join([us0,'forum.php?mod=forumdisplay&fid=',str(fid)])
        for xc in range(0,pn9):
            uss=''.join([us2,'&page=',str(xc)]);#print(uss);
            urls.append(uss)
    #
    return urls


def zmul_getHtm(df9,urls,xfun,nwork=10,ftg='tmp/mul100.csv'):
    #u100=list(map(lambda x : ''.join([us0,x,'/']),x100))
    pool=ThreadPoolExecutor(max_workers = nwork)
    #pool=ProcessPoolExecutor(max_workers = nwork)
    xsubs = [pool.submit(xfun,uss) for uss in urls]
    #
    for xsub in as_completed(xsubs):
        [df,coinsgn,uss]=xsub.result(timeout=60);
        print('@mx.u9:',uss,'@dn',len(df.index))
        if len(df.index)>0:
            df9=df9.append(df,ignore_index=True)
    #
    if ftg!='':zdat.fdf_wr(df9,ftg,kn=9,fgFlt=True)
    #
    return df9        
    #    
#------bs4.xxx
def bs_get_ktag(tag):
    #return tag.has_attr('isend')
    #print('k',fb_get_ktag_kstr)
    return tag.has_attr(zsys.bs_get_ktag_kstr)
    