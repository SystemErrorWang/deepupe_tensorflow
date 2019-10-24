update 2019.10.24:

The writer open-sourced official version, so this version could be abondoned. In fact the bilateral filter part cannot get reasonable results(because i don't use C++ and dont's know how to make the official version in HDRNet), and there are also other bugs in my code.


In fact, I tried to modify Deep Guided Filter and got pretty good results in color adjusting/HDR task.(https://github.com/wuhuikai/DeepGuidedFilter) Its inplementation only involves official tensorflow ops and does not neet to add self-defined ops. I am sorry that the code could not be published because it's relevant to my job. But i believe it's easy to modify by your self.

______________________________________________________________________________________________________________________________


update 2019.07.11:

I believe codes in this repo cannot get you any reasonale result. It may be the codes themselves (bilateral upsample), or the dataset(they are not strictly aligned, so pixel-wise losses are not suitable). Recently I tried deep guided filter with re-process dataset, and got much better result. I will update my code once I have time, maybe in this repo, or open a new repo.

______________________________________________________________________________________________________________________________

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

______________________________________________________________________________________________________________________________


I guess this code can run on recent tensorflow versions without main problems but you may need to modify the dataloader part, as I am using post-processed hdr_burst dataset with photos arranged in my own way.

There are some differences bewteen my implementation and the original paper:

In the paper, VGG19 is used as feature extractor but here I use mobilenetv2 the pre-trained model need input data to be normalized to (-1, 1), but according to my experiments, this will result in bad visual quality, so you may consider not use the pre-trained weight or even try to use another network (see in network.py)

The smoothness loss is also different: I kept getting nan with the logarithmic operation so I deleted that part, the left part could be seen as total-variation loss (tv-loss)

also, the bilateral slice op is borrowed from this repo: https://github.com/dragonkao730/BilateralGrid/blob/master/bilaterial_grid.py

I am still training the model, and will upload the pre-trained weight if I got visual satisfying result.(There are some problems with these codes now, i am trying to modify them and will update later)
