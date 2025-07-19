#! /bin/bash
docker cp `pwd`/$1 c0:/workspace/$1
docker cp `pwd`/$1 c1:/workspace/$1
sleep 1
gnome-terminal --window -- bash -c "docker exec c0 conda run -n batchcrypt  python $1 --rank=0 --size=2;echo Output of 0; exec bash"
sleep 1
gnome-terminal --window -- bash -c "docker exec c1 conda run -n batchcrypt  python $1 --rank=1 --size=2;echo Output of 1; exec bash"
