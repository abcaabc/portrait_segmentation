1.
模型文件已经训练并保存，可以直接运行web.py文件，选择并上传人像图片进行便可以得到分割结果。
如果想要自己体验一下模型训练的过程，可以参考以下俩点：

    1.训练集与验证集
    在train文件夹下的img_train/images中放入训练集图片，masks中放入对应掩码
    在train文件夹下的img_ver/images中放入验证集图片，masks中放入对应掩码

    2.训练模型
    运行文件train文件夹下的my_train.py文件即可训练模型，最后模型文件会保存至model文件夹中
    （如果有需要可以调节my_train.py中的训练参数，或是优化神经网络resUnet34.py，抑或是重构数据集my_dataset.py）

2.
上传的图片会保存在文件夹upload_images中，已经提供了测试图片test.jpg
