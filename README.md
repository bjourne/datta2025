# Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks
Codes for Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks

## Usage

Please first change the variable "DIR" at File ".Preprocess\getdataloader.py", line 9 to your own dataset directory

Train model with QCFS-Layer 

```bash
python main.py train --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --l=QUANTIZATION_STEP
```
Test accuracy in ann mode or snn mode

```bash
python main.py test --bs=BATACHSIZE --model={vgg16, resnet18} --data={cifar10, cifar100, imagenet} --id=YOUR_MODEL_NAME --mode={ann, snn} --t=SIMULATION_TIME
```

For inference, run the following

For l=3, 7, 15, 31, 255, 65535 and models=vgg16, resnet34, mobilenetv2 and hoyer_decay=0,2e-9

python main.py test --bs=BATACHSIZE --model={values above} --data=imagenet --id=YOUR_MODEL_NAME --mode=ann --l={values above} --hoyer_decay={values above}

Only for mobilenetv2, do

python main.py test --bs=BATACHSIZE --model=mobilenetv2 --data=imagenet --id=YOUR_MODEL_NAME --mode=snn --l={values above}

Note down the sparsity values shown in the console for vgg16 and resnet34 and hoyer_decay=0 and hoyer_decay=2e-9.

For comparison with ANN2SNN_COS, do cd ANN2SNN_COS, then run this command

python main.py --dataset ImageNet --load_model_name {model path without .pth} --net_arch {vgg16,resnet34} --batchsize 64 --CUDA_VISIBLE_DEVICE {device_id} --l 7  --presim_len 4 --sim_len 8

Only note down the sparsity values shown in the console fror net_arch=vgg16 and resnet34.
