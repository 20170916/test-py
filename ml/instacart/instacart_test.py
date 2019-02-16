#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: lo time:2019-02-16

import pandas as pd
from sklearn.decomposition import PCA

# 读取四张表数据
prior = pd.read_csv("./order_products__prior.csv")
products = pd.read_csv("./products.csv")
orders = pd.read_csv("orders.csv")
aisles = pd.read_csv("aisles.csv")

# 合并四张表到一张表中（用户-物品类别）
_mg = pd.merge(prior, products, on=['product_id', 'product_id'])
_mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])

# 查看前10个
mt.head(10)

# 交叉表（特殊的分组工具）
cross = pd.crosstab(mt["user_id"], mt["aisle"])

cross.head(10)

# 主成分分析
pca = PCA(n_components=0.9)
data = pca.fit_transform(cross)
data
data.shape
