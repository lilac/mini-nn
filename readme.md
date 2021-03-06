## 关于

这是一个小巧的多层神经网络实现。它不依赖第三方数据学习库。


## 安装

下载后切换到项目根目录

1. 安装python虚拟环境（可选）
```
virtualenv venv
```
2. 安装依赖
```
pip install -r requrements.txt
```

## 使用

确保先切换到`src`目录

所有命令行操作都是由`util.py`脚本实现。它支持子命令：`t`用于训练，`e`用于评估和预测。使用`-h`参数可以获取详细的参数帮助信息。

### 训练
子命令：t

参数信息
* -n --epochs 训练代数
* -b --batch 批次大小
* -e --eta 学习率
* -l --sizes 每个网络层的神经元个数
* --validate 在训练的每一代评估模型在验证集上的准备率
* -o --output 模型保存路径

#### 示例
```
python util.py t --validate -n 10 -b 10 -l 784 56 28 10
```
上面命令训练一个包含两个隐层的神经网络。每层含有的神经元个数依次为：784，56，28，10。

### 评估
子命令：e

参数
* -d --dir 测试图片目录
* 必须参数 模型文件路经

#### 示例
```
python utils.py e -d ../samples ../models/best.pkl.gzip
```
此命令会预测出`samples`目录下所有图片中的数字。

如果省去`-d`参数，则它会默认评估mnist数据集中测试集的准确率。
```
python utils.py e ../models/best.pkl.gzip
```

## 结果
目前训练得最优模型保存在文件`models/best.pkl.gzip`中。它的网络结构为
```
输入层  中间层  输出层
784-->56->28->10
```
此模型在mnist测试集中的错误率为5.85％。