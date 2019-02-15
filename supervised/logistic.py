# -*- coding: utf-8 -*-
# @Author  : itswcg
# @File    : logistic.py
# @Time    : 19-2-15 上午9:35
# @Blog    : https://blog.itswcg.com
# @github  : https://github.com/itswcg

"""
概率：已知一个模型和参数，推数据
统计：根据一堆数据，去预测模型和参数

logistic 用于二分类问题
回归：有一些点，我们用一条直线对这些点进行拟合（该线为最佳拟合直线），这个拟合过程称作回归
利用回归分类思想：根据现有数据对分类边界建立回归公式，以此进行分类
sigmoid函数： O(z) = 1 / (1+e^-z) 阶跃函数


"""
