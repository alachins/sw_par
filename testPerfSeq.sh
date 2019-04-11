gcc -o run main.c -O3 -D_GNU_SOURCE -D_SEQUENTIAL -D_MODE_1 
./run > rep1.txt

gcc -o run main.c -O3 -lpthread -D_GNU_SOURCE -D_SEQUENTIAL -D_MODE_$1 
./run > rep$1.txt

tkdiff rep1.txt rep$1.txt
