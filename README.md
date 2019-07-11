# deepupe_tensorflow

This is an unofficial implementation of cvpr2019 paper "Underexposed Photo Enhancement using Deep Illumination Estimation"
-------------
paper url: http://jiaya.me/papers/photoenhance_cvpr19.pdf

My testing environment is as below:

ubuntu 16.04

tensorflow-gpu 1.12

cuda 9.0

cudnn 7.3.1

pytorch 1.0(i used pytorch dataloader)


update 2019.07.10:

I believe the codes in this repo will not get reasonale result, maybe the code (bilateral upsample), maybe the dataset(they are not strictly aligned, so pixel-wise loss will learn a mess). Recently I tried deep guided filter with re-process dataset, and got much better result. I will update my code once I have time, maybe in this repo, or open a new repo.


I guess this code can run on recent tensorflow versions without main problems but you may need to modify the dataloader part, as I am using post-processed hdr_burst dataset with photos arranged in my own way.

There are some differences bewteen my implementation and the original paper:

In the paper, VGG19 is used as feature extractor but here I use mobilenetv2 the pre-trained model need input data to be normalized to (-1, 1), but according to my experiments, this will result in bad visual quality, so you may consider not use the pre-trained weight or even try to use another network (see in network.py)

The smoothness loss is also different: I kept getting nan with the logarithmic operation so I deleted that part, the left part could be seen as total-variation loss (tv-loss)

also, the bilateral slice op is borrowed from this repo: https://github.com/dragonkao730/BilateralGrid/blob/master/bilaterial_grid.py

I am still training the model, and will upload the pre-trained weight if I got visual satisfying result.(There are some problems with these codes now, i am trying to modify them and will update later)
