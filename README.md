# Context-aware Neural Machine Translation with Mini-batch Embedding

This respository includes the example scripts of [the following paper](https://www.aclweb.org/anthology/2021.eacl-main.214/):
```
Context-aware Neural Machine Translation with Mini-batch Embedding
Makoto Morishita, Jun Suzuki, Tomoharu Iwata, Masaaki Nagata
https://www.aclweb.org/anthology/2021.eacl-main.214/
```

## Requirements
- Python 3
- [PyTorch](https://pytorch.org/)
- [sentencepiece](https://github.com/google/sentencepiece)
- sacrebleu
```bash
pip install "sacrebleu[ja]"
```
- NVIDIA GPU with CUDA


## Data preprocessing
This will download the corpora and preprocess the files.
```bash
$ cd ./corpus
$ ./process.sh
```

## Build fairseq
In order to run fairseq, you need to build.
```bash
$ cd ./tools/fairseq_doc
$ pip install --editable .
```


## Training
The training scripts are available in `./en-ja/`.
You may need to change the `PROJECT_DIR` variable in the scripts.

This is an example of training a MBE enc model.
```bash
$ cd ./en-ja
$ nohup train_model_mbe_enc.sh 1 &> train_model_mbe_enc.log &
```


## Contact
Please send an issue on GitHub or contact us by email.

NTT Communication Science Laboratories  
Makoto Morishita  
makoto.morishita.gr -a- hco.ntt.co.jp  
