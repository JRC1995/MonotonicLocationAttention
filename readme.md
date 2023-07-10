## Official Code for Monotonic Location Attention for Length Generalization (ICML 2023)

[ArXiv Link](https://arxiv.org/abs/2305.20019)

### Credits:
* ROPE code in models/utils is taken from: HuggingFace Library (transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding)
* lookup data generated using: https://github.com/i-machine-think/machine-tasks

### Requirements
* torch==1.10.0
* tqdm==4.62.3
* nltk==3.6.5  
* jsonlines==2.0.0
* tensorflow-datasets==4.5.2

### Data Setup
Put [them](https://github.com/i-machine-think/machine-tasks/tree/master/SCAN/length_split) in ```dataset/scan/length```.

### Preprocess/Generate
Go to preprocess/ and run the files (those are also the file to check into if you want to develop on our synthetic data generators).
We also release the exact splits that we used [here](https://drive.google.com/file/d/1Ov0tP4GVlIvLNcVdknxosG8WBmAlOmfC/view?usp=sharing). 

### Train
```python trian.py --model=[insert model name] -- dataset=[insert dataset name] --times=[insert total runs] --device=[insert device name] --test=[True/False]```

(see argparser.py for options). 

### Mapping task names in code (left) to names in paper (right):
* copy == Copy
* rc == Reverse Copy
* llt == Lookup
* rllt == Reverse Lookup
* fpc == ReCopy
* rpc == Reverse ReCopy
* fgc == Inv ReCopy
* rgc == Inv Reverse ReCopy
* posretrieve == PosRetrieve
* dedup == DeDupe
* scan_length = SCAN (length split)
* cfq_length == CFQ (length split)
 
### Mapping model names in code (left) to names in paper (right):
* BiGRU == Content Attention
* BiGRUrel == Relative Attention
* BiGRUrel_mixdir == Bi-Relative Attention
* BiGRUrope_mixdir == Bi-ROPE Attention
* BiGRU_locattn_simple == LocAttn S
* BiGRU_locattn == LocAttn
* BiGRU_mixattn_simple == Mix LocAttn S
* BiGRU_mixattn == Mix LocAttn
* BiGRU_mixattn_simplePR == Mix LocAttn S PR
* BiGRU_OneStep == OneStepAttn
* BiGRU_MixOneStep == Mix OneStepAttn
* BiGRU_MixOneStepPR == Mix OneStepAttn PR
* BiGRU_MonoAttn == MonoAttn
* BiGRU_MixMonoAttn == Mix MonoAttn
* BiGRU_MixMonoAttn PR == Mix MonoAttn PR
* BiGRU_RMonoAttn == RMonoAttn
* BiGRU_MixRMonoAttn == Mix RMonoAttn
* BiGRU_MixRMonoAttn PR == Mix RMonoAttn PR
* BiGRU_AblationStep == OneStepAttn - Step 2
* BiGRU_SoftStairStep == OneStepAttn - Step 3
* BiGRU_FreeStep == OneStepAttn - Sigmoid

### Citation

```
@InProceedings{pmlr-v202-ray-chowdhury23b,
  title = 	 {Monotonic Location Attention for Length Generalization},
  author =       {Ray Chowdhury, Jishnu and Caragea, Cornelia},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {28792--28808},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/ray-chowdhury23b/ray-chowdhury23b.pdf},
  url = 	 {https://proceedings.mlr.press/v202/ray-chowdhury23b.html},
  abstract = 	 {We explore different ways to utilize position-based cross-attention in seq2seq networks to enable length generalization in algorithmic tasks. We show that a simple approach of interpolating the original and reversed encoded representations combined with relative attention allows near-perfect length generalization for both forward and reverse lookup tasks or copy tasks that had been generally hard to tackle. We also devise harder diagnostic tasks where the relative distance of the ideal attention position varies with timestep. In such settings, the simple interpolation trick with relative attention is not sufficient. We introduce novel variants of location attention building on top of Dubois et al. (2020) to address the new diagnostic tasks. We also show the benefits of our approaches for length generalization in SCAN (Lake &amp; Baroni, 2018) and CFQ (Keysers et al.,2020). Our code is available on GitHub.}
}
```
Contact the associated github email for any question or issue. 



