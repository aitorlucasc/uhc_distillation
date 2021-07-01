# Deep Learning for content-based indexing of TV programs

This repository is part of our final thesis from the MSc in Fundamental Principles of Data Science.

You can find all about the master in: https://mat.ub.edu/sciencedata/

## About the project: 
In recent years, deep neural networks have been successful in a lot of tasks in bothindustry and academia due to its scalability to mange large volumes of data andmodel parameters. Unfortunately, creating those large models and use their predic-tions can be computationally expensive to deploy on devices with limited resources.There  is  a  TV  channel  called  TV3  that  wants  to  improve  its  recommendation  en-gine without the mentioned impediments.   In that thesis,  we aim to solve part ofthat problem by using YOLO and Places to detect objects and scenes respectively,and build a smaller model able to learn from them and extract frame objects andscenes by itself.  To do it, we have analyzed in depthHeterogeneous Classifiers (HC),that ensemble models with some different classes in a smaller model using a convexoptimization approach. As HCs do not handle an scenario where classes differ com-pletely between models, which is the TV3 case, we have implemented the smallermodel following a label prediction approach by using RMSE and we have evaluatedthe model with ranking metrics as we have faced an unsupervised problem.
## About the authors:
Noel Rabella: https://github.com/Noel-bs/

Aitor Lucas: https://github.com/aitorlucasc/

We are two students finishing the master, previously we studied a BSc in Telecommunications Engineering in Universitat Pompeu Fabra.


## Target audiences
This project is addressed to upper-tier undergraduate and beginning graduate students from technical disciplines. Moreover, the thesis is also addressed to professional audiences following continuous education short courses and to researchers from diverse areas following self-study courses.

Basic skills in computer science, mathematics and statistics are required. Python and Pytorch are the basics of this project.

## Link to the thesis
To be published.

### BibTex reference format for citation for the Code
```
@misc{uhcRef,
title={Deep Learning for content-based indexing of TV programs},
url={https://github.com/aitorlucasc/uhc_distillation/},
note={GitHub repository with al the code of the thesis.},
author={Aitor Lucas, Noel Rabella},
  year={2021}
}
```
### BibTex reference format for citation for the report of the Master's Thesis

```
@misc{uhcRefReport,
title={Deep Learning for content-based indexing of TV programs},
url={https://github.com/aitorlucasc/uhc_distillation/},
note={Report of the Master's Thesis: Deep Learning for content-based indexing of TV programs.},
author={Aitor Lucas, Noel Rabella},
  year={2021}
}
```
