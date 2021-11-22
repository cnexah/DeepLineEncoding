# [Deep Line Encoding for Monocular 3D Object Detection and Depth Prediction](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0299.html)
>In autonomous driving scenarios, straight lines and vanishing points are important cues for single-image depth perception. In this paper, we propose the deep line encoding to make better use of the line information in scenes. More specifically, we transform potential lines into parameter space through the deep Hough transform. The aggregation of features along a line encodes the semantics of the entire line, whereas the voting location indicates the algebraic parameters. For efficiency, we further propose the novel line pooling to select and encode the most important lines in scenes. With deep line encoding, we advance the state-of-the-art on KITTI single-image 3D object detection and depth prediction benchmarks.

## Note
We evaluate deep line encoding on two tasks **seperately**: monocular 3d object detection and depth prediction.  
For the monocular 3d object detection task, we use only the ImageNet (for pretrain) and the **official training set** to train the model.   
## Setup
Please follow [VisualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D) to setup the environment.  
## Monocular 3D Object Detection
Only support a single GPU  
Please enter the folder  
```
cd object_detection
```
Please modify the path in config/config.py (for train and val) and config/config_test.py(for train and test)  
Precompute:
```
python scripts/imdb_precompute_3d.py --config=config/config.py # for train and validation
python scripts/imdb_precompute_test.py --config=config/config_test.py # for train and test
```
Train:
```
python train.py --config=config/config.py
```
Eval:
```
python scripts/eval.py --config=config/config.py --checkpoint_path=model.pth
```
## Depth Prediction
Please enter the folder
```
cd depth_prediction
```
Train:  
```
python scripts/train.py --config=config/config.py # for single GPU
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/train.py --config=config/config.py --world_size=$NUM_GPUS # for multiple GPUs
```
Eval:
```
python scripts/eval.py --config=config/config.py --checkpoint_path=model.pth
```
## Related Code
Our code is based on [VisualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D) and [Deep-Hough-Transform-Line-Priors](https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors)

## Papers
```
@inproceedings{ce21dle,
 author = {Ce Liu and Shuhang Gu and Luc Van Gool and Radu Timofte},
 booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
 title = {Deep Line Encoding for Monocular 3D Object Detection and Depth Prediction},
 year = {2021}
}
```
```
@ARTICLE{9327478,
  author={Y. {Liu} and Y. {Yuan} and M. {Liu}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Ground-aware Monocular 3D Object Detection for Autonomous Driving}, 
  year={2021},
  doi={10.1109/LRA.2021.3052442}}
```
```
@article{lin2020deep,
  title={Deep Hough-Transform Line Priors},
  author={Lin, Yancong and Pintea, Silvia L and van Gemert, Jan C},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```
