# 模型一：Baseline
### 相关脚本

baseline_train.py、baseline_model.py、baseline_test.py

### 参数

`--evol` 	超参数自动优化

`--train、--valid、--in_feature` 	数据集路径和数据维度

### 目前效果

28维数据集（未归一化）——24轮——0.7065

&nbsp;

# 模型二：ResNet

### 说明

用ResNet网络预测

### 相关脚本

res_train.py、res_model、res_test.py

### 参数

`--evol` 	超参数自动优化

`--device ` 	可选cpu、cuda

`--train、--valid、--in_feature` 	数据集路径和数据维度

### 目前效果

28维数据集（未归一化）——273轮——0.8196

&nbsp;

# 模型三：LCNet

### 说明

### 相关脚本

LCNet_train.py、LCNet_model.py、LCNet_test.py

### 参数

`--device ` 	可选cpu、cuda

`--train、--valid、--in_feature` 	数据集路径和数据维度

### 目前效果

33维数据集（未归一化）——131轮——0.8226

&nbsp;

# 模型四：SVM

### 说明

使用sklearn库调用SVM的现有模型进行预测

### 相关脚本

SVM.py

### 数据集

`data/ML/`	进行了数据归一化、空值填充-1、正负样本均衡的处理

### 参数

`--clf`	选择分类器：SVC、LinearSVC

`--kernel`	选择核函数：rbf、poly

`--C`	错误项的惩罚系数

`--auto`	自动把C从0~50以0.1为步长依次拟合预测并保存结果到`auto_result.csv` 中

### 目前效果

SVC—poly—degree=2—C=11—0.7656



&nbsp;

# 模型五：Decision Tree
### 代码说明

- ``decision_tree.py``: 运行决策树train和test的文件
- ``config/tree_decistion_config.py``: 决策树的相关配置
- train后会生成一个pdf和一个模型文件，pdf是决策树的可视化pdf;模型文件保存在treeCheckpoints下
- test时需要指定具体的模型文件
### 运行说明

- 具体配置参见`config/tree_decision_config.py`
- 训练代码示例:
    ```bash
    python decision_tree.py
  ```
- 测试代码示例:
    ```bash
    python decision_tree.py --test True --model treeCheckpoints/10-20-14-18.pkl
    ```

### 目前效果

效果一般，待尝试

&nbsp;

# 模型六：Random Forest

### 代码说明

- ``forest.py``: 运行决策树train和test的文件
- ``config/forest_config.py``: 决策树的相关配置
- train后的模型文件保存在treeCheckpoints下
- test时需要指定具体的模型文件，输出到当前运行目录
### 运行说明

- 具体配置参见`config/tree_decision_config.py`
- 训练代码示例:
    ```bash
    python decision_tree.py
  ```
- 测试代码示例:
    ```bash
    python decision_tree.py --test True --model treeCheckpoints/10-20-14-18.pkl
    ```

&nbsp;

# 模型七：AdaBoost

### 说明

使用sklearn库调用AdaBoost的现有模型进行预测

### 相关脚本

AdaBoost.py

### 数据集

`data/ML/`	进行了数据归一化、空值填充-1、正负样本均衡的处理

### 参数

`--base_estimator`	用于集成的基础分类器

 `--n_estimators`	集成的分类器数量

`--lr`	每次训练的学习率

`--feature`	数据维度

`--auto`	尚未完善

### 目前效果

使加权结果提升一个点

&nbsp;

# 模型结果的加权

### 1. `utils/vote.py`
- 脚本`utils/vote.py`
- 此脚本要求文件目录如下:
  
  ----history
  
  &nbsp;&nbsp;&nbsp;&nbsp;----model_name1

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.abcd.txt

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.abcd.pt

  &nbsp;&nbsp;&nbsp;&nbsp;----model_name2

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.efgh.txt

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----xxxxx_xxx_0.efgh.pt
  
- 注意不要更改`weighted`文件夹名

- 注意输出的txt结尾是四位小数的格式

&nbsp;

### 2. `test.py`

**直接对三种模型进行加权测试**
- 注意目前没有baseline的模型(主要是没有训好的baseline提交的分数)，如果有了之后，
  要和其他文件的命名方式一样并且放在history/Fake1DAttention下
- 模型的各种参数设置在`config/test_config.py`中，可自己查阅（目前baseline模型的config不正确，所以不要直接跑！！！！）
- 每次运行后，各个模型也会输出一个结果（在history下对应目录，反复运行会覆盖），总加权后的结果也会输出（也就是说目前是4个）

&nbsp;

### 3. `vote_lrk.py`

对当前五个模型的最优参数训出模型的预测结果进行加权，结果保存在 `history/weighted/vote_lrk.txt`

##### 加权方式

&emsp;&emsp;把各模型在对应正负样本平衡的验证集valid上测试的“1”准确率和“0”准确率作为其在测试集test的“1”和“0”的可信度，对test中每个样本由各模型得出的分类结果结合其可信度来加权获得最终分类结果。

##### 目前效果

&emsp;&emsp;Base0.6581_Res0.8196_LC0.8226_DT0.8142——0.8409，比各模型单独都要好

&nbsp;

# 数据集说明

### 当前模型对应数据集

| 模型     | train                     | valid                     | test                     |
| -------- | ------------------------- | ------------------------- | ------------------------ |
| baseline | unmodified/train.csv      | unmodified/valid.csv      | unmodified/test_a.csv    |
| ResNet   | 同上                      | 同上                      | 同上                     |
| LCNet    | 33_dimension/train.csv    | 33_dimension/valid.csv    | 33_dimension/test.csv    |
| SVC_rbf  | ML/33_dimension/train.csv | ML/33_dimension/valid.csv | ML/33_dimension/test.csv |
| SVC_poly | ML/28_dimension/train.csv | ML/28_dimension/valid.csv | ML/28_dimension/test.csv |

### 数据分析

所有35维数据的相关性分析：

`/data/original/pearmon`、`/data/original/spearman` 分别保存了皮尔曼系数和斯皮尔曼系数

### 改动内容

从user_track中提取出总登录次数 total_day、工作日登录比例 work_day_rate、周末登录比例 weekend_day_rate、平均第一次登陆时间 avg_early_hour、平均最后一次登录时间avg_last_hour，现在完整数据见 data\original\all_info.csv

### 目录说明

`33_dimension/` 	加入了user_track里的5个维度

`original/`	原始数据

`unmodified/`	baseline的数据集

`ML/`	机器学习算法的数据集，其中 `spearman_selected`下是与label的斯皮尔曼相关系数大于0.1的量组成的数据集

`balanced`	后缀的是正负样本平衡后的数据集

`0.5valid`	前缀的是正负样本比为1：1的验证集，没有此前缀的是6：4的验证集

### 随机切分数据

`utils/random_data.py` 用于随机切分数据，如果要在模型训练测试时使用,可以直接调用其中的`generate_new_data()`函数,生成的新数据
会在`data/random_data/`下

&nbsp;

# History目录

用于记录已提交文件测试效果及对应网络参数文件等

`test_a`	为初赛历史记录

`test_b`	为复赛历史记录



`/Fake1DAttention` 下为只使用全连接，文件命名格式：`数据维度(是否归一化)_训练轮数_测试分数`

`/ResNet` 下为使用残差网络，文件命名格式：`数据维度(是否归一化)_训练轮数_测试分数`

`/LCNet` 下为使用LCNet（conv），文件命名格式：`数据维度(是否归一化)_训练轮数_测试分数`

`/ML`	下为使用机器学习算法预测的结果及分数，以及auto模式下的模型效果

`/weighted`	下为几个模型预测结果加权后的结果及分数

&nbsp;

# TODO

### lrk

###### 一期

​		数据预处理：transforms（已完成归一化）、删除部分维度（已完成相关性分析）

​		尝试不同：loss_fn、optimizer（均已完成，效果不佳）

​		超参数优化加到ResNet网络中（已完成），试试效果（效果一般）

###### 二期

​		数据集中正负样本平衡化（效果卓越，大幅提高了DicisionTree和SVM的效果）

​		尝试机器学习模型分类：LGBM、SVM（效果尚可，0.76左右）、随机森林

​		按与label斯皮尔曼相关性高的标签重组数据集，训练SVM（效果一般）

###### 三期

​		用在平衡样本数据集上的预测准确率作为加权对模型预测结果进行融合（效果较为优秀）

​		加权脚本自动化，对不同参数下的相同模型预测结果进行加权

###### 四期

​		分析test_a的label

​		重新整理所有形式的数据集，加入test_a_label

​		处理data目录，去除冗杂，保持简洁

​		各模型在新数据集上重新训练

### xyf

### lc
