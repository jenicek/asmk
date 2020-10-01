# Python ASMK (Aggregated Selective Match Kernels)

Official ASMK Python implementation for our [ECCV 2020 paper](https://arxiv.org/abs/2007.13172):

```
@InProceedings{TJ20,
  author      = "Giorgos Tolias and Tomas Jenicek and Ond\v{r}ej Chum}",
  title       = "Learning and aggregating deep local descriptors for instance-level recognition",
  booktitle   = "European Conference on Computer Vision",
  year        = "2020"
}
```

## Running the Code

1. Install the requirements (`faiss-cpu` for cpu-only setup)

```
pip3 install numpy faiss-gpu
```


2. Build C library for your Python version

```
python3 setup.py build_ext --inplace
rm -r build
```


3. Download `cirtorch` and add it to your `PYTHONPATH`

```
wget "https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/master.zip"
unzip master.zip
rm master.zip
export PYTHONPATH=${PYTHONPATH}:$(realpath cnnimageretrieval-pytorch-master)
```


4. Run `examples/demo_how.py` giving it any `.yaml` parameter file from `examples/params/*.yml`


### Reproducing ECCV 2020 Results

Reproducing results from **Table 2.**

- R18<sub>how</sub> (n = 1000): &nbsp; `examples/demo_how.py eccv20_how_r18_5` &ensp; _ROxf (M): 75.1, RPar (M): 79.4_
