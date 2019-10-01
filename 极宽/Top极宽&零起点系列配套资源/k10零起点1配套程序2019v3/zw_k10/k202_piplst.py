# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import pip

#=============

print('''新版本pip，get_installed_distributions()函数接口改了
      本案例取消
      开源项目，函数API接口，参数变化，属于很正常的现象
      包括，谷歌的TensorFlow，intel的openCV，还有pandas数据分析模块，
      每次大的版本升级，都会有个别函数API接口变化，
      这种因为版本变化，引发的程序代码冲突，称为：版本冲突
      所以，使用开源软件，要养成多动手搜索/查看最新版本的软件文档/函数接口餐宿
      ''')

#--------------------

'''
plst=pip.get_installed_distributions();
print(plst[10])

df=pd.DataFrame();
df['<name>']=plst;
print(df.tail())
fss="tmp\\m100.csv";print("\n"+fss)
df.to_csv(fss,index=False)

'''