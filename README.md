# TF_Face_MultiGPU

## Requirement
TensorFlow r1.8 or above.

NCCL2 or above

## Features
1. Basic train & feature extracting pipeline for deep face recognition.
2. Stable and efficient Multi-GPU training support.
3. TensorFlow Dataset API support for efficient I/O from Caffe style lists.
4. Random mirror/rotation/brightness/contrast/hue(RGB only)/saturation(RGB only) data augmentation.
5. Network support: *ResNeXt*, *MobileNet*, *ShuffleNet*, *SENet*, *SphereFaceNet*, *LightCNN*(coming soon).
6. Loss support: *center loss*, *triplet loss*, *A-softmax loss*\*, *Coco Loss*(coming soon).
7. Automatic GPU selection. (utils/gpu_select.py)

Results on mainstream face recognition benchmarks are coming soon.

\* As far as we know, our code is the first A-softmax loss implementation in TensorFlow. [@shangwenxiang](https://github.com/shangwenxiang) claims to reproduce the LFW accuracy **99.4%** on SphereFaceNet-20 with our implementation. It exceed the results for original implementation.

## Usage
For training:

	python train.py --num_gpus=4 \
	--model_name='Your model name.' \
	--net_name='Your net name' \
	--data_list_path='Your caffe-style list path.' \
	--batch_size=512 \
	...
	...

For feature extraction:

	python evaluate.py \
	--model_name='Your model name. \
	--net_name='Your net name' \
	--fea_name='Your feature name' \
	--data_list_path='Your caffe-style list path.' \
	--batch_size=200 \
	...
	...

## Reference
Comming soon...
