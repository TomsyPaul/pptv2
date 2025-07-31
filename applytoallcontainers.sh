#! /bin/bash
j=0
for i in `cat hostips`
do 
echo "Trying contianer c$j on host $i.." 
ssh tomsy@$i 'docker exec ' c$j $1  < /dev/null
((j++))
done
