# Inferring Contour Sketches from Images
## ENGN2560 - Computer Vision
### Final Project (*May, 2019*)


#### Xingchen Ming xingchen_ming@brown.edu 
#### Ming Xu ming_xu1@brown.edu 
#### Geng Yang   geng_yang@brown.edu

<p align="center"><img alt="Teaser" src="doc/teaser.jpg"></p>

# Dataset
## NOTE: Please download and extract the dataset under directory `PhotoSketch/`

### https://drive.google.com/open?id=1ajNGbYSSxWZyCT3X4qlga7maUz6UZ3nD

# Setting up on Brown CCV

1. Load Anaconda3-5.2.0

``` 
module load anaconda/3-5.2.0
```


2. One-line installation (with Conda environments)
```
conda env create -f environment.yml
```

3. Activate the environment
```
source activate sketch
```

# Running Instructions
## NOTE: All srcipts should be executed under directory `PhotoSketch/`
## Train model
```
sbatch cuda.sh
```
## Test model
1. Request a GPU node
```
interact -n 16 -m 16g -q gpu -g 1
```
2. Run the test script
```
sh scripts/test_pretrained.sh
```



## Citation
If you use the code or the data for your research, please cite the paper:

```
@article{LIPS2019,
  title={Photo-Sketching: Inferring Contour Drawings from Images},
  author={Li, Mengtian and Lin, Zhe and M\v ech, Radom\'ir and and Yumer, Ersin and Ramanan, Deva},
  journal={WACV},
  year={2019}
}
```

## Acknowledgement
This code is based on an old version of [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/).

