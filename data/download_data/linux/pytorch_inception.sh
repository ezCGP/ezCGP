# to calc FID score in SimGAN
DATADIR=../../datasets/pytorch_models
mkdir -p $DATADIR
for PTH in 'inception_v3_google-1a9a5a14.pth' 'inception_v3_google-0cc3c7bd.pth'
do
	wget -O $DATADIR/$PTH -c https://download.pytorch.org/models/$PTH --no-check-certificate
done