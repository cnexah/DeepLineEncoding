# Deep Line Encoding for Monocular 3D Object Detection and Depth Prediction
>In autonomous driving scenarios, straight lines and vanishing points are important cues for single-image depth perception. In this paper, we propose the deep line encoding to make better use of the line information in scenes. More specifically, we transform potential lines into parameter space through the deep Hough transform. The aggregation of features along a line encodes the semantics of the entire line, whereas the voting location indicates the algebraic parameters. For efficiency, we further propose the novel line pooling to select and encode the most important lines in scenes. With deep line encoding, we advance the state-of-the-art on KITTI single-image 3D object detection and depth prediction benchmarks.

## Setup
Please follow [VisualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D) to setup the environment.  
Baiscally:
```
pip install -r requirement.txt
./make.sh
```
## Monocular 3D Object Detection
Only support a single GPU  
Precompute:
```
python scripts/imdb_precompute_3d.py --config=config.py # for train and validation
python scripts/imdb_precompute_test.py --config=config.py # for test
```
Train:
```
python train.py --config=config/config.py
```
Eval:
```
python scripts/eval.py --config=config.py --checkpoint_path=model.pth --split_to_test="validation"
``` 
