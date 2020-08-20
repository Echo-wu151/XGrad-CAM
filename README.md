# XGrad-CAM implementation in Pytorch 

This is the official pytorch implementation of the paper:
### [Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs](https://arxiv.org/pdf/2008.02312v3)

to be presented at **BMVC 2020 (Oral presentation)**,
<br>
Authors:
<br>
Ruigang Fu,
[Qingyong Hu](https://qingyonghu.github.io/)
Xiaohu Dong,
[Yulan Guo](http://yulanguo.me/),
Yinghui Gao and
Biao Li,
<br>

**[[Paper](https://arxiv.org/abs/2008.02312)] [[Blog](https://zhuanlan.zhihu.com/p/175994533)]** <br />

----------

### XGrad-cam.py
XGrad-CAM is a CNN visualization method, try to explain why classification CNNs predict what they predict. It is class-discriminative, efficient and able to highlight the regions belonging to the objects of interest.

<img src="https://github.com/Fu0511/XGrad-CAM/blob/master/examples/XGrad-CAM.png" width="70%">

The main difference between XGrad-CAM and Grad-CAM locates at line 116 - line120:
#####  XGrad-CAM
`X_weights = np.sum(grads_val[0, :] * target, axis=(1, 2))`

`X_weights = X_weights / (np.sum(target, axis=(1, 2)) + 1e-6)`
#####  Grad-CAM 
`weights = np.mean(grads_val, axis=(2, 3))[0, :]`

Usage: `python XGrad-cam.py --image-path <path_to_image> --target-index <class_of_interest>` for CPU computation, add `--use-cuda` for GPU acceleration.

Example: `python XGrad-CAM.py --image-path ./examples/ILSVRC2012_val_00000077.JPEG --target-index 159 --use-cuda`

Output: `class of interest:  n02087394 Rhodesian ridgeback`

Results:

![Grad-CAM](https://github.com/Fu0511/XGrad-CAM/blob/master/examples/cam.jpg) ![XGrad-CAM](https://github.com/Fu0511/XGrad-CAM/blob/master/examples/X_cam.jpg)

left is Grad-CAM, right is XGrad-CAM

----------

### Proof_verify.py
This is a simple script of experimental proof for our statement that given an arbitrary layer in ReLU-CNNs, there
exists a specific equation between the class score and the feature maps of the layer (Eq.(5) in our paper).

Usage: `python Proof_verify.py --image-path <path_to_image> --target-index <class_of_interest>`

For any class of interest, the result will show that `class_score-gradients*feature-bias_term=0`

----------

These codes are based on https://github.com/jacobgil/pytorch-grad-cam.
Thanks to the author Jacob Gildenblat for the beautiful original code.

If you find our work useful in your research, please consider citing:
```
@inproceedings{fu2020axiom,
  title={Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs},
  author={Fu, Ruigang and Hu, Qingyong and Dong, Xiaohu and Guo, Yulan and Gao, Yinghui and Li, Biao},
  booktitle={British Machine Vision Conference},
  year={2020}
}
```

