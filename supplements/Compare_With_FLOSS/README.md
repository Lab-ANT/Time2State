# Note
There may be a bug in the sklearn package. The result may overflow, resulting in negative ARI values, when calculating ARI for extreme long sequence. This error only happens when evaluating the results on the raw PAMAP2 dataset, and does not happen on the 4x downsampled dataset.

Please manually change the type of variables in the **adjusted_rand_score** function of local sklearn package to solve this problem.

```python
tn = np.float64(tn)
fp = np.float64(fp)
fn = np.float64(fn)
tp = np.float64(tp)
```

# Matlab Version
The version of matlab used is 2020b.

# Prepare Python Environment

Step 1: prepare python environment  
We use python 3.9 and anaconda to manage the environment.
```bash
conda create -n [env name] python=3.9
conda activate [env name]
```
Step 2: Install TSpy  
TSpy is under developing, we can access it on the dev branch.
```bash
git clone git@github.com:Lab-ANT/TSpy  
cd TSpy  
git checkout dev  
python setup.py install
cd ..
rm -rf TSpy
```  
Step 3: Install other packages  
```bash
pip install -r requirements.txt
```

# Download Data
The data used for evaluation, the figures and intermediate products are archived [here](https://drive.google.com/drive/folders/1Yr5buUN6QNK4NcvnPI6JxAlUD7sX8eJH?usp=share_link) for easy available. 

# How to run FLOSS
Please follow the following steps to run FLOSS+TSKMeans:

1. use **convert_to_csv.py** to convert data format for convient evaluation. Downsampling rate is specified in this script.
2. use **run_FLOSS.m** to run FLOSS on all datasets, except for UCR-SEG.
3. use **flossScore.m** to run FLOSS on UCR-SEG.
4. use **extract_seg_pos_from_FLOSS_output.py** to extract segment boundaries found by FLOSS
5. use **cluster_segs.py** to conduct clustering step for the segment results of FLOSS.
6. rename the output_FLOSS folder, e.g., output_FLOSS-dtw or output_FLOSS-euclidean
7. use **plot_result_of_FLOSS.py** to plot results of FLOSS on all datasets. The result figures are saved in figs/
8. use **evaluate_FLOSS.py** to calculate ARI and NMI for FLOSS+TSKMeans.

Please remember to adjust the corresponding parameters in the script when switching distance metrics.

# How to run Time2State
Please run these scripts in the following order:

1. use **convert_to_csv2.py** to convert data format for Time2State. The difference between **convert_to_csv.py** and **convert_to_csv2.py** lies in the naming format, and there is no other differences.
2. run **exp_of_Time2State.py** to execute Time2State on all datasets.
3. run **evaluate_Time2State.py** to evaluate the results of Time2State.
4. run **plot_result_of_Time2Staet.py** to plot the results of Time2State.