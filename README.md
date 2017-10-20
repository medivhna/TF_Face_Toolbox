# TF_Face_MultiGPU

## Requirement
TensorFlow r1.3 or above.
NCCL 1.3.2

## Features
1. Basic train & feature extracting pipeline for deep face recognition.
2. Stable and efficient Multi-GPU training support.
3. Random mirror/rotation/brightness/contrast/hue(RGB only)/saturation(RGB only) data augmentation.
4. Network support: *ResNeXt*\*, *MobileNet*, *ShuffleNet*, *SENet*, *SphereFaceNet*.
5. Loss support: *center loss*, *triplet loss*, *A-softmax loss*(coming soon), *Coco Loss*(coming soon).
6. LFW validation (Coming soon...)

\*It is difficult to implement efficient group convolution for ResNeXt on TensorFlow, wait for cudnn7 support.

## Usage
For training:

	CUDA_VISIBLE_DEVICES=0,1,2,3 \
	python train.py --num_gpus=4 \
	--model_name='Your model name.' \
	--data_list_path='Your caffe-style list path.' \
	--batch_size=512 \
	...
	...

For feature extraction:

	CUDA_VISIBLE_DEVICES=0 \
	python evaluate.py \
	--model_name='Your model name. \
	--fea_name='Your feature name' \
	--data_list_path='Your caffe-style list path.' \
	--batch_size=200 \
	...
	...

## Reference
Comming soon...