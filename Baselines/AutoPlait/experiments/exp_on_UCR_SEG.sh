#!/bin/sh
# Created by Chengyu on 2021/12/5.

# compile.
cd ./src
make cleanall
make
cd ..

# Configuration.
data_source="../../data/"
INPUTDIR=$data_source"UCR-SEG/UCR_AutoPlait/"
OUTDIR="output/"

outdir=$OUTDIR"_out_UCR_SEG/"
dblist=$INPUTDIR"list"
n=32  # data size
d=1  # dimension

rm -rf $outdir
mkdir -p $outdir

for (( i=1; i<=$n; i++ ))
do
  # start=$(date +"%s.%N")
  output=$outdir"dat"$i"/"
  mkdir -p $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}'# $dblist > $input
  ./src/autoplait $d $input $output
  # end=$(date +"%s.%N")
  # my_array[$i]=$(timediff $start $end)
done