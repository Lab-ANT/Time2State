#!/bin/sh
# Created by Chengyu on 2021/12/5.

# compile.
cd ./src
make cleanall
make
cd ..

# Configuration.
data_source="../../data/"
OUTDIR="output/"

# exp_on_UCR_SEG
# INPUTDIR=$data_source"UCR-SEG/UCR_AutoPlait/"
# outdir=$OUTDIR"_out_UCR_SEG/"
# dblist=$INPUTDIR"list"
# n=32  # data size
# d=1  # dimension

# exp_on_synthetic
INPUTDIR=$data_source"synthetic_data_for_Autoplait/"
outdir=$OUTDIR"_out_synthetic/"
dblist=$INPUTDIR"list"
n=100  # data size
d=4  # dimension

# USC-HAD
# INPUTDIR=$data_source"USC-HAD_for_AutoPlait/"
# outdir=$OUTDIR"_out_USC-HAD/"
# dblist=$INPUTDIR"list"
# n=70  # data size
# d=6  # dimension

# exp_on_MoCap
# INPUTDIR=$data_source"MoCap/4d/"
# outdir=$OUTDIR"_out_MoCap/"
# dblist=$INPUTDIR"../list"
# n=9  # data size
# d=4  # dimension

# exp_on_ActRecTut
# INPUTDIR=$data_source"ActRecTut/data_for_AutoPlait/"
# outdir=$OUTDIR"_out_ActRecTut/"
# dblist=$INPUTDIR"list"
# n=2  # data size
# d=10  # dimension

# # exp_on_PAMAP2
# INPUTDIR=$data_source"PAMAP2/data_for_AutoPlait/"
# outdir=$OUTDIR"_out_PAMAP2/"
# dblist=$INPUTDIR"list"
# n=10  # data size
# d=9  # dimension

rm -rf $outdir
mkdir -p $outdir

for (( i=1; i<=$n; i++ ))
do
  start=$(date +"%s.%N")
  output=$outdir"dat"$i"/"
  mkdir -p $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}'# $dblist > $input
  ./src/autoplait $d $input $output
  end=$(date +"%s.%N")
done