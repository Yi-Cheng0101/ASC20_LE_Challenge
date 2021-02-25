#!/usr/bin/zsh
# CONFIG
FILES=$1"/pytorch_model"
EPOCH_NUM=$2
COUNT=`expr $EPOCH_NUM - 1`

rm -r debug-exp

for i in {0..$COUNT}
do 
	echo "processing $FILES$i.bin"
	mv $FILES$i.bin $FILES.bin
	./eval.sh $1 $3
	mv $FILES.bin $FILES$i.bin
	echo "done eval $i"
done

if [ -d $1"/debug-exp" ]; then
	rm -r $1"/debug-exp"
fi

cp -r ./debug-exp $1
