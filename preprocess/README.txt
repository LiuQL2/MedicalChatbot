数据预处理，从原始的文件得到可以进行对话的user goal文件，总共包括以下几个步骤：
但是前提需要手动整理好哪些是top疾病，且top疾病中口语表达symptom和归一化symptom之间的对应关系，即top_disease_symptom_aligned.json
文件。
1. 运行top_disease.py 文件，里面定义好需要抽取的几种疾病名称；
2. 运行match_disease.py， 从包含主诉症状的文件中抽取出前几种疾病的主诉内容和相应的症状。
3. 运行extract_symptom.py文件，分别从主诉文本、对话内容中抽取疾病症状，得到的症状都是口语化表达形式；