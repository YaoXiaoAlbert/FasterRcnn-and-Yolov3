# FasterRcnn-and-Yolov3

## yolov3
模型比较小，训好的直接在文件夹里了
主要是在这个框架下（https://github.com/ultralytics/yolov3）修改配置文件
下载下来运行：

python train.py --data data/voc.yaml --cfg data/hyps/hyp.scratch-low.yaml

即可训练，配置文件在./yolov3/data/hyps/hyp.scratch-low.yaml修改，且可以在训练时修以如下方式修改超参：

python train.py --data data/voc.yaml --cfg data/hyps/hyp.scratch-low.yaml --epochs 50

图片放在data/images下并运行这行命令即可测试模型：

python detect.py --source data/images --weights runs/train/exp9/weights/best.pt --conf 0.25


## faster-rcnn
解压voc2007trainval在同目录后运行train.py
运行eval.py可以对图片进行检测，修改eval.py中的图片路径即可
如果不训练的话可以从https://pan.baidu.com/s/1ByTwCvefnRAkd9APqD1BeA?pwd=nadm 下载训练好的模型，放在其train.py目录下即可
