#! /bin/bash
#cd ~/mydfl
#git pull https://tomsypaul@github.com/TomsyPaul/mydfl.git 
#read -p "Enter Worldsize " worldsize
worldsize=$5
#set hostips
head -n $worldsize hostipsall > hostips

#generate layouts
root=`python3 treegen.py --n=$worldsize`

##generate secrets (n-1)
#>secrets
#secretsum=0
#for((i=1;i<$worldsize;i++))
#do
#thisrandom=`echo $RANDOM/100000 | bc -l|cut -c 1-8`
#echo "$thisrandom" >> secrets
#secretsum=`echo $secretsum+$thisrandom | bc -l|cut -c 1-8`
#done
##generate secrets (n-th)
#echo "-$secretsum" >> secrets


#generate partition_sizes
>partition_sizes
for((i=0;i<$worldsize;i++))
do
#common=`echo 1.0/$worldsize | bc -l`
common=`echo 1.0/16 | bc -l`
echo -n "$common, ">>partition_sizes 
done

#generate keys
>keys
keycount=`echo "$worldsize/4" |bc`
for((i=0;i<$keycount;i++))
do
echo "$RANDOM" >> keys
done

#set files to upload
>files-to-upload
echo layout >> files-to-upload
echo layout-up >> files-to-upload
echo layout-down >> files-to-upload
#echo secrets >> files-to-upload
echo keys >> files-to-upload
echo run.py >> files-to-upload
echo partition_sizes >> files-to-upload
echo ftl >> files-to-upload
#rest of the process
mkdir -p results
coding=$1
epochs=$2
averager=$3
K=$4
runid=`date +'%Y-%m-%d_%H-%M-%S'`
if [ $coding == 'N' ]
then
   bash applytoall.sh 'docker stop' c
   bash applytoall.sh 'docker rm' c
   
#  bash applytoall.sh 'docker image rm 9446917617/mydfl-image:pytorch-docker'
   bash copytoall.sh Dockerfile
   bash copytoall.sh run.py
   bash copytoall.sh layout-up
   bash copytoall.sh layout-down
   bash copytoall.sh layout
   bash copytoall.sh keys
   bash copytoall.sh secrets
   bash buildall.sh
   read
   bash setup.sh $5
   bash applytoall.sh 'docker start' c
else
echo $worldsize-$averager-$epochs-$runid>>logslist
bash runscript.sh $coding $epochs $averager $K $runid $root
read
echo "$worldsize,$2,$3,$K,$runid" > "results/$worldsize-$averager-$epochs-$runid"
echo -e "******************\n" >> "results/$worldsize-$averager-$epochs-$runid"
bash applytoallcontainers.sh "cat /logs/$worldsize-$averager-$epochs-$runid;echo" >> "results/$worldsize-$averager-$epochs-$runid"
echo -e "Result..\n"
cat results/$worldsize-$averager-$epochs-$runid
echo "$worldsize,$2,$3,$K,$runid" >> "results/summary"

grep TIME results/$worldsize-$averager-$epochs-$runid | cut -d"," -f4 | awk '{ sum += $1; n++ } END { if (n > 0) print "Average time taken = " sum / n "\n"; }' >> results/summary

grep BYTES results/$worldsize-$averager-$epochs-$runid | cut -d"," -f6 | awk '{ sum += $1; n++ } END { if (n > 0) print "Average bytes sent = " sum / n "\n"; }' >> results/summary

grep MESSAGES results/$worldsize-$averager-$epochs-$runid | cut -d"," -f8 | awk '{ sum += $1; n++ } END { if (n > 0) print "Average Messages sent = " sum / n "\n"; }' >> results/summary

echo -e "Average Loss\n" >> "results/summary"

for((i=0;i<$epochs;i++))
do 
grep "epoch,$i" results/$worldsize-$averager-$epochs-$runid | cut -d"," -f5 | awk '{ sum += $1; n++ } END { if (n > 0) print "'$i' = " sum / n ; }' >> results/summary
done
echo "" >> "results/summary"
bash close-all-terminals.sh
fi
