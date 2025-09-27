# Graph Learning on Text-Attributed Graphs: Bias Mitigation and Semantic Smoothing via LLMs

## 1.Environment Setup with Conda

conda create -n BSLGL python=3.8 -y

conda activate BSLGL

**CPU**

conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

**GPU**

conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -y

conda install numpy pandas scikit-learn -y

conda install -c conda-forge sentence-transformers chardet requests -y

conda install -c anaconda gensim nltk pytz -y

conda install -c pyg pyg -y

python -c "import nltk; nltk.download('punkt')"

## 2.Download
The following related files need to be downloaded.

**Orig TAGs datasets:**  
[dataset](https://drive.google.com/drive/folders/158wnv1zp2xOX2fKCUeLrzpxrpKRkJJzO?usp=drive_link)

**BSLGL_processed texts:**  
[enhanced_texts](https://drive.google.com/drive/folders/1e8WMWOM46jhUhMQqwn0cJPCZSIv0J-lN?usp=drive_link)

## 3.Train
**Example：**

python -m code.trainGNN  --dataset cora --gnn_model_name SAGE --gnn_train_feature_type enhanced

Specific parameter modifications can be made by selecting the corresponding options in the **BSLGL/code/config.py** file according to needs.
