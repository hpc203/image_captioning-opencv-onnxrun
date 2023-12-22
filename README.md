本套程序里的模型使用传统CNN做图像描述的，通俗来讲就是输入一张图片，输出一句描述图片里的内容的文字，简称图片生成文字。

包含encoder和decoder两个模块，其中decoder模块的输入张量是二维的，opencv-dnn在输入这样形状的张量时推理报错，因此decoder模块
使用onnxruntime做推理引擎。
onnx文件在百度云盘，
链接：https://pan.baidu.com/s/1mVI7_ey_Iu2r_X9DEwAbtg 
提取码：vbo9

训练源码在https://github.com/ruotianluo/ImageCaptioning.pytorch


现在火热的多模态大模型clip，连接了图像和语义文字两个领域的。
这使得clip天生就适合做图片描述的，但是模型太大，在我的个人笔记本电脑运行会占用大量内存，因此我占时没有编写用clip做图片生成文字的程序。
