python neural_style.py ^
train ^
--epochs 2 ^
--batch-size 2 ^
--dataset .data/train_images ^
--style-image .data/style_image.jpg ^
--save-model-dir .data/model ^
--checkpoint-model-dir .data/model/checkpoints ^
--cuda 1 ^
--seed 42 ^
--content-weight 1e5 ^
--style-weight 1e10 ^
--lr 1e-3 ^
--log-interval 500 ^
--checkpoint-interval 2000