# install shared Anaconda environment following: https://medium.com/@pjptech/installing-anaconda-for-multiple-users-650b2a6666c6
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3*.sh -u
#chmod -R go-w /opt/anaconda
#chmod -R go+rX /opt/anaconda
source ~/.bashrc
cd ~
git clone https://github.com/ezCGP/ezCGP
cd ezCGP
git checkout 2020S-gpu
sudo ln -s $HOME/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh
conda create -n ezCGP python=3.6 anaconda -y
conda activate ezCGPP
conda config --env --add channels menpo
conda config --env --add channels conda-forge
conda install --file requirements.txt -y
source ~/.bashrc
pip3 install horovod
