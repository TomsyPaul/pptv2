#! /bin/bash
for i in `ps -A | grep bash | tail -n 4 | tr -s " " | cut -d" " -f2`
do
kill -9 "$i"
done
