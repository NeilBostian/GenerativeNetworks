## Fast Neural Style (modified)
A copy / modified version of [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style)

See [usage.md](./usage.md)

### Training Setup
[Download COCO 2014 Training Images (83K/13GB)](http://images.cocodataset.org/zips/train2014.zip) ([Parent Site](http://cocodataset.org/#download)) and place in relative directory ./.data/train_images/sub_dir

_Note the directory train_images should contain sub_dir which contains the actual images, due to how the pytorch Dataset loads files_

### Python Environment
    conda create --name neural_style python=3.7.4
    conda activate neural_style
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    conda install pillow=6.2.0