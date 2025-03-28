#! /bin/bash
i=0
while read ip
do
 if [ ! -z $ip ]
 then
   if [ ! -z $2 ]
   then
      container="c$i"
   else 
      container=""
   fi      
   ssh tomsy@$ip $1 $container </dev/null&
   ((i++))     	
#   echo "id=$i, ip=$ip"
 fi
done < hostips
