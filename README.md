# Time2State

This is the repository for the paper entitled "Time2Seg: An Unsupervised Framework for Inferring the Latent State in Time Series Data".

# Installation

Running Time2State requires the installation of other packages.

```python
# Clone Time2State
git clone git@github.com:Lab-ANT/Time2State

# Install TSpy
git clone git@github.com:Lab-ANT/TSpy
cd TSpy && python setup.py install && pip install requirements.txt && cd ..

# Install TSAGen
git clone https://github.com/Lab-ANT/TSAGen
cd TSAGen && python setup.py install && pip install requirements.txt && cd ..

# Install other packages
cd Time2State && pip install requirements.txt
```

# Data Preparation

Download the dataset and put them in the following position.

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

# How to Run

run the *.py files in ./experiments directly
