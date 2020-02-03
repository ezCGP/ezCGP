# install shared Anaconda environment following: https://medium.com/@pjptech/installing-anaconda-for-multiple-users-650b2a6666c6
sudo rm /etc/profile.d/conda.sh
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
mkdir -f /home/anaconda
bash Anaconda3*.sh -u -b -p /home/anaconda
#chmod -R go-w /opt/anaconda
#chmod -R go+rX /opt/anaconda
cd /
source ~/.bashrc
git clone --single-branch --branch 2020S-gpu https://github.com/ezCGP/ezCGP
cd /ezCGP
git checkout 2020S-gpu
sudo ln -s /home/anaconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
. /etc/profile.d/conda.sh
conda create -n ezCGP python=3.6 anaconda -y
conda activate ezCGP
conda config --env --add channels menpo
conda config --env --add channels conda-forge
ls
conda install --file requirements.txt -y
conda install -c qiqiao horovod
ln -s /home/Anaconda/bin/conda /usr/local/bin/conda
