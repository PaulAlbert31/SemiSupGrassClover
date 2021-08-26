#Train on the synthetic images and predict labels from the unlabeled images
CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet34 --lr 0.007 --workers 12 --epochs 1 --gpu-ids 0 --eval-interval 1 --dataset irish --batch-size 4 --checkname resnet34irish --base-size 1024 --crop-size 1024 --loss-type ce --use-balanced-weights
CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet34 --lr 0.007 --workers 12 --epochs 1 --gpu-ids 0 --eval-interval 1 --dataset irish --batch-size 4 --checkname resnet34irish --base-size 1024 --crop-size 1024 --loss-type ce --use-balanced-weights --resume "run/irish/resnet34irish/checkpoint_best.pth.tar" --predict "../samples/irish/images/"
mv preds preds_irish_lab

CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet34 --lr 0.007 --workers 12 --epochs 1 --gpu-ids 0 --eval-interval 1 --dataset irish --batch-size 4 --checkname resnet34irish --base-size 1024 --crop-size 1024 --loss-type ce --use-balanced-weights --resume "run/irish/resnet34irish/checkpoint_best.pth.tar" --predict "../samples/irish_ext/images/"
mv preds preds_irish_unlab

python predictbiomass_irish.py --masks preds_irish_lab --gt ../samples/irish/gt.csv --test preds_irish_unlab
mv biomasscomposition.csv ../samples/irish_ext/train.csv

mv train.csv ../samples/irish/train.csv
mv val.csv ../samples/irish/val.csv

CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet34 --lr 0.007 --workers 12 --epochs 1 --gpu-ids 0 --eval-interval 1 --dataset danish --batch-size 4 --checkname resnet34danish --base-size 1024 --crop-size 1024 --loss-type ce --use-balanced-weights

CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet34 --lr 0.007 --workers 12 --epochs 1 --gpu-ids 0 --eval-interval 1 --dataset danish --batch-size 4 --checkname resnet34danish --base-size 1024 --crop-size 1024 --loss-type ce --use-balanced-weights --resume "run/danish/resnet34danish/checkpoint_best.pth.tar" --predict "../samples/danish/images/"
mv preds preds_danish_lab

CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet34 --lr 0.007 --workers 12 --epochs 1 --gpu-ids 0 --eval-interval 1 --dataset danish --batch-size 4 --checkname resnet34danish --base-size 1024 --crop-size 1024 --loss-type ce --use-balanced-weights --resume "run/danish/resnet34danish/checkpoint_best.pth.tar" --predict "../samples/danish_ext/images"

mv preds preds_danish_unlab

python predictbiomass_danish.py --masks preds_danish_lab --gt ../samples/danish/gt.csv --test preds_danish_unlab

mv biomasscomposition.csv ../samples/danish_ext/train.csv

mv train.csv ../samples/danish/train.csv
mv val.csv ../samples/danish/val.csv
