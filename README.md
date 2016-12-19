# What's this
Implementation of Weighted Residuals for Very Deep Networks Depth (WResNet) by chainer  

# Dependencies

    git clone https://github.com/nutszebra/weighted_residual_net.git
    cd weighted_residual_net
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for some parts.  

* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Drop probability  
The linear decay is used. P_0 is 1 and P_L is 0.5.


# Cifar10 result
| network                  | depth   | total accuracy (%) |
|:-------------------------|---------|-------------------:|
| WResNet-d [[1]][Paper]   | 1192    | 95.3               |
| my implementation        | 1192    | soon               |

<img src="https://github.com/nutszebra/weighted_residual_net/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/weighted_residual_net/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References  
Weighted Residuals for Very Deep Networks [[1]][Paper]

[paper]: https://arxiv.org/abs/1605.08831 "Paper"
