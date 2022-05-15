# 三维点云生成 & 人体动作识别模型

## 配置环境

[PyTorch3D](https://pytorch3d.readthedocs.io/en/latest/overview.html)这个库提供了Ball Query算法、FPS算法、点云体素化的高性能CUDA实现，可以提高数十倍效率。但是它对PyTorch的Python的版本要求特别严格，经测试下面这行命令可以安装成功（Linux和CUDA 11.4）。

```bash
conda create -n MachineLearning python=3.9 pytorch=1.9.1 cudatoolkit fvcore iopath pytorch3d -c pytorch3d -c pytorch -c fvcore -c iopath -c conda-forge
```

还有其他一些库：

```bash
pip install matplotlib numpy sklearn scipy tqdm
```

## 数据集

* Pantomime数据集：http://dx.doi.org/10.5281/zenodo.4459969
* RadHAR数据集：https://github.com/nesl/RadHAR
* PeopleWalking数据集（雷达原始数据）：https://doi.org/10.1016/j.dib.2020.105996

数据集放在data文件夹内，文件结构大概这样子：

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

第一次读取数据集的时候会对数据集预处理，预处理的结果保存为data.pickle，之后只读取data.pickle

所以可以直接用我预处理之后的数据，这样就可以不用下载数据集了：

https://1drv.ms/u/s!Ap0_tHPGTLjfhv1cZZpv_iQyyNnExA?e=aBzV1G

## 训练日志

log文件夹里记录了训练过程中会产生的所有数据：
* 每个epoch的训练集和测试集的accuracy和loss
* 训练参数
* 效果最好的模型，最后的模型
* 混淆矩阵

我的毕设论文中的所有65个实验的训练日志：

https://1drv.ms/u/s!Ap0_tHPGTLjfhv1fUuyTU2nmQwvCQg?e=0VBFdz

## 点云生成的可视化图片和视频

PointCloudGeneration.ipynb生成的图片和视频保存在fig文件夹中，用来做答辩PPT和论文插图，包括：
* Range FFT结果（图片）
* 静态物体去除后的结果（图片）
* 距离-方位角热力图（视频）
* CFAR算法的结果（视频）
* DBSCAN算法的结果（视频）
* 带速度值的距离-方位角热力图（视频）

整理之后图片和视频共享链接：

https://1drv.ms/u/s!Ap0_tHPGTLjfh4A2tp6LUW7s-1fsVA?e=pShZbn

## 实验代码

每次实验的代码都可以通过git保存的历史代码看到，commit message中以training config结尾的都是那次训练使用的代码
