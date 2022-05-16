This is the code of this paper: [Task-oriented Dialogue System for Automatic Diagnosis](http://www.aclweb.org/anthology/P18-2033).

A Medical-domain Dialogue System for Diseases Identification.
Both the symptoms and the diseases are treated as slots as in the conventional dialogue systems.
There is only one agent in the dialogue system, who is in charge of both symptom selection and disease prediction.

# Data
Our dataset is available [here](https://github.com/LiuQL2/MedicalChatbot/blob/develop/acl2018-mds.zip) or can be found at the homepage of [Prof. Wei](http://www.sdspeople.fudan.edu.cn/zywei/). Please read the README.txt in the zip file and our paper for the details about our dataset.

# How to use
1. Downloading our [dataset](http://www.sdspeople.fudan.edu.cn/zywei/data/acl2018-mds.zip).
2. Putting the dataset in a directory and pointing the path of the dataset in src/dialogue_system/run/run.py
3. Using the following command to see how to run this code, i.e., each parameter in the code.
```
cd src/dialogue_system/run
python run.py --help
```

# Cite
```
@inproceedings{wei2018task,
  title={Task-oriented Dialogue System for Automatic Diagnosis},
  author={Liu, Qianlong and Wei, Zhongyu and Peng, Baolin and Tou, Huaixiao and Chen, Ting and Huang, Xuanjing and Wong, Kam-Fai and Dai, Xiangying},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  volume={2},
  pages={201--207},
  year={2018}
}
```
