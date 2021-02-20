# https://github.com/EN10/CIFAR
DATADIR=../../datasets/cifar10
mkdir -p $DATADIR
wget -O $DATADIR/cifar-10-python.tar.gz -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -C $DATADIR/ -xvzf $DATADIR/cifar-10-python.tar.gz
rm ./$DATADIR/cifar-10-python.tar.gz