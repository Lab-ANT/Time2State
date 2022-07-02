# Time2State

This is the repository for the paper entitled "Time2State: An Unsupervised Framework for Inferring the Latent State in Time Series Data".

# Installation

Running Time2State requires the installation of other packages.

```python
# Install TSpy
git clone git@github.com:Lab-ANT/TSpy
cd TSpy
pip install requirements.txt
python setup.py install
cd ..

# Clone Time2State
git clone git@github.com:Lab-ANT/Time2State
cd Time2State
pip install requirements.txt
```

# Data Preparation

Download [PAMAP2](http://www.pamap.org/demo.html) and [USC-HAD](http://sipi.usc.edu/had/) and put them in the following position. 

```
.
├── data
│   ├── ActRecTut
│   ├── synthetic_data_for_segmentation
│   ├── MoCap
│   ├── PAMAP2
│   │   ├── Protocol
│   │   │   ├── subject101.dat
│   │   │   ├── ...
│   ├── USC-HAD
│   │   ├── Subject1
│   │   ├── Subject2
│   │   ├── ...
```

Once the data is placed correctly, run the following script.
```
python ./scripts/prepare_data.py
```

# How to Run

run the *.py files in ./experiments directly
