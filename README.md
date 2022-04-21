# Time2Seg

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