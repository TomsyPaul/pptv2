#! /bin/bash
size=`cat hostips | wc -l`
for i in `ps -A | grep bash | tail -n $size | tr -s " " | cut -d" " -f2`
do
kill -9 "$i"
done
