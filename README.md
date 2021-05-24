# 赛题介绍
芒果TV-第二届“马栏山杯”国际音视频算法大赛-防盗链
随着业务的发展，芒果的视频内容也深受网友的喜欢，不少视频网站和应用开始盗播芒果的视频内容，盗链网站不经过芒果TV的前端系统，跳过广告播放，且消耗大量的服务器、带宽资源，直接给公司带来了巨大的经济损失，因此防盗链在日常运营中显得尤为重要。如何从海量日志中识别出盗链行为，并进行有效拦截，也是防盗链工作的核心。

&nbsp;

## 模型介绍
 - 对原始特征进行LabelEncoder编码保存为feather文件
 - 构建 group_feature、count_feature、count_time_feature、cumcount_time_feature、cumcount_feature、cumcountratio_feature、diff_feature、category_encoders_feature，对每类特征加原始特征进行LightGBM建模得到特征重要性
 - 汇总各类特征重要性后分别取前50、100、150、200、250个特征进行LightGBM建模，提交200个特征结果B榜得分0.5174

&nbsp;

## 代码

```python
python preprocess.py # 原始数据文件放入/data文件夹中，处理原始数据

python feature_engineering.py # 特征工程，单个特征保存为feather文件

python main.py # 模型训练，按日期排序，最后两天数据作为线下验证集数据
```

