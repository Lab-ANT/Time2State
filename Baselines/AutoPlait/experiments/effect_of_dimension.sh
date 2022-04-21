#!/bin/sh
# Created by Chengyu on 2021/12/5.

function timediff() {
    # time format:date +"%s.%N", such as 1502758855.907197692
    start_time=$1
    end_time=$2
    
    start_s=${start_time%.*}
    start_nanos=${start_time#*.}
    end_s=${end_time%.*}
    end_nanos=${end_time#*.}
    
    if [ "$end_nanos" -lt "$start_nanos" ];then
        end_s=$(( 10#$end_s - 1 ))
        end_nanos=$(( 10#$end_nanos + 10**9 ))
    fi

    time=$(( 10#$end_s - 10#$start_s )).`printf "%03d\n" $(( (10#$end_nanos - 10#$start_nanos)/10**6 ))`
    echo $time
}

# compile.
cd ./src
make cleanall
make
cd ..

# Configuration.
data_source="../../data/"
INPUTDIR=$data_source"effect_of_dim/"
OUTDIR="output/"

outdir=$OUTDIR"effect_of_dim/"
dblist=$INPUTDIR"list"
n=20  # data size
# d=4  # dimension

rm -rf $outdir
mkdir -p $outdir

for (( i=1; i<=$n; i++ ))
do
  start=$(date +"%s.%N")
  output=$outdir"dat"$i"/"
  mkdir -p $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}'# $dblist > $input
  echo $input
  ./src/autoplait $i $input $output
  end=$(date +"%s.%N")
  my_array[$i]=$(timediff $start $end)
  echo $(timediff $start $end)
done

echo "time list:"${my_array[*]}