# M2DC
M2DC: A Meta-Learning Framework for Generalizable Diagnostic Classification of Major Depressive Disorder

This code implements the group DRO algorithm from the following paper:

[M2DC: A Meta-Learning Framework for Generalizable Diagnostic Classification of Major Depressive Disorder](https://ieeexplore.ieee.org/abstract/document/10680596)

# Abstract:
Psychiatric diseases are bringing heavy burdens for both individual health and social stability. The accurate and timely diagnosis of the diseases is essential for effective treatment and intervention. Thanks to the rapid development of brain imaging technology and machine learning algorithms, diagnostic classification of psychiatric diseases can be achieved based on brain images. However, due to divergences in scanning machines or parameters, the generalization capability of diagnostic classification models has always been an issue. We propose Meta-learning with Meta batch normalization and Distance Constraint (M 2 DC) for training diagnostic classification models. The framework can simulate the train-test domain shift situation and promote intra-class cohesion, as well as inter-class separation, which can lead to clearer classification margins and more generalizable models. To better encode dynamic brain graphs, we propose a concatenated spatiotemporal attention graph isomorphism network (CSTAGIN) as the backbone. The network is trained for the diagnostic classification of major depressive disorder (MDD) based on multi-site brain graphs. Extensive experiments on brain images from over 3261 subjects show that models trained by M 2 DC achieve the best performance on cross-site diagnostic classification tasks compared to various contemporary domain generalization methods and SOTA studies. The proposed M 2 DC is by far the first framework for multi-source closed-set domain generalizable training of diagnostic classification models for MDD and the trained models can be applied to reliable auxiliary diagnosis on novel data.

# Prerequisites:
- python 3.8
- os
- random
- torch
- numpy
- tqdm
- einops
- torchvision
- time
- pathlib
- sklearn
- csv
- copy
- argparse
- datetime
