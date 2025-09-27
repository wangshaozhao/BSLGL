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

## 2.Download TAGs

[dataset]https://drive.google.com/drive/folders/158wnv1zp2xOX2fKCUeLrzpxrpKRkJJzO?usp=drive_link)
