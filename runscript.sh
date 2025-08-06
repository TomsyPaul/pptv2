#! /bin/bash
#for i in `wc -l hostips`
size=`wc -l hostips | cut -d" " -f1`
coding=$1
epochs=$2
averager=$3
K=$4
runid=$5
i=0
while  read ip
do
 if [ ! -z $ip ]
 then
   if [ $coding == 'Y' ]
   then
   while read filename
   do
      scp $filename tomsy@$ip:mydfl
      ssh -n tomsy@$ip docker cp /home/tomsy/mydfl/$filename c$i:/workspace/$filename
   done < files-to-upload      
   fi   
 ((i++))     	  
 fi
done < hostips

i=0
while  read ip
do
 if [ ! -z $ip ]
 then
   gnome-terminal --window -- bash -c "ssh -n tomsy@$ip docker exec c$i python run.py --rank=$i --size=$size --epochs=$epochs --averager=$averager --K=$K --runid=$runid; echo Output of $i; exec bash"   
 ((i++))     	
 fi
done < hostips

