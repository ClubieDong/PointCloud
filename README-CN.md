# 三维点云生成 & 人体动作识别模型

## 简介

本仓库是我的本科毕业论文[《基于毫米波雷达的人体动作识别系统》](https://github.com/ClubieDong/PointCloud/blob/main/%E5%9F%BA%E4%BA%8E%E6%AF%AB%E7%B1%B3%E6%B3%A2%E9%9B%B7%E8%BE%BE%E7%9A%84%E4%BA%BA%E4%BD%93%E5%8A%A8%E4%BD%9C%E8%AF%86%E5%88%AB%E7%B3%BB%E7%BB%9F.pdf)的实验代码。

`PointCloudGeneration.ipynb`中实现了利用雷达原始数据生成三维点云的算法，包括距离FFT、计算距离-方位角热力图、多普勒FFT等步骤，生成的每个点包含距离、角度、信号强度、速度等五个维度的信息。

`PointCloudClassification`文件夹中实现了一套由点云特征提取层、时间序列处理层和全连接分类层组成的机器学习模型架构，每个部分可以独立变换组合。在点云特征提取层中使用基于体素的三维卷积和PCA方式、PointNet和PointNet++来提取空间域特征；在时间序列处理层中使用RNN、GRU、LSTM来提取时间域特征。

## 环境配置

[PyTorch3D](https://pytorch3d.readthedocs.io/en/latest/overview.html)这个库提供了Ball Query算法、FPS算法、点云体素化算法的高性能CUDA实现，可以提高数十倍效率。但是它对PyTorch和Python的版本要求特别严格，经测试下面这行命令可以安装成功（在Linux上使用CUDA 11.4）。

```bash
conda create -n MachineLearning python=3.9 pytorch=1.9.1 cudatoolkit fvcore iopath pytorch3d -c pytorch3d -c pytorch -c fvcore -c iopath -c conda-forge
```

安装其他使用到的库：

```bash
pip install matplotlib numpy sklearn scipy tqdm
```

## 数据集

* Pantomime数据集：http://dx.doi.org/10.5281/zenodo.4459969
* RadHAR数据集：https://github.com/nesl/RadHAR
* PeopleWalking数据集（雷达原始数据）：https://doi.org/10.1016/j.dib.2020.105996

数据集放在`data`文件夹内，文件结构大概这样子：

```
/data
  /Pantomime
    /primary_exp
      ...
    /supp_exp
      ...
    data.pickle
  /PeopleWalking
    1.mat
    2.mat
    ...
  /RadHAR
    /Test
      ...
      data.pickle
    /Train
      ...
      data.pickle
/PointCloudClassification
  ...
PointCloudGeneration.ipynb
README.md
```

第一次读取数据集的时候会对数据集预处理，预处理后的结果保存为`data.pickle`，之后只会读取`data.pickle`。

所以你可以直接用我提供的预处理之后的数据，这样就可以不用下载数据集了：

https://1drv.ms/u/s!Ap0_tHPGTLjfhv1cZZpv_iQyyNnExA?e=aBzV1G

## 训练日志

`log`文件夹里记录了训练过程中会产生的所有数据：
* 每个epoch的训练集和测试集的accuracy和loss
* 训练参数
* 效果最好的模型，最后的模型
* 混淆矩阵

我的本科毕业论文中的所有65个实验的训练日志：

https://1drv.ms/u/s!Ap0_tHPGTLjfhv1fUuyTU2nmQwvCQg?e=0VBFdz

## 点云生成的可视化图片和视频

`PointCloudGeneration.ipynb`生成的图片和视频保存在`fig`文件夹中，用来做答辩PPT和论文插图，包括：
* Range FFT结果（图片）
* 静态物体去除后的结果（图片）
* 距离-方位角热力图（视频）
* CFAR算法的结果（视频）
* DBSCAN算法的结果（视频）
* 带速度值的距离-方位角热力图（视频）

整理后图片和视频共享链接：

https://1drv.ms/u/s!Ap0_tHPGTLjfh4A2tp6LUW7s-1fsVA?e=pShZbn

## 实验代码

每次实验的代码都可以通过git保存的历史代码看到，commit message中以`training config`结尾的都是那次训练使用的代码。
