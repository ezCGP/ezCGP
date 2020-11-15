# honestly, its prob best to do all this by hand and not through shell script

# https://github.gatech.edu/emade/emade
echo ""
echo "----- INSTALL GIT LFS FIRST ----"
echo "https://docs.github.com/en/free-pro-team@latest/github/managing-large-files/installing-git-large-file-storage"
echo ""
echo "----- MAKE SURE YOU ARE ON GATECH VPN!!! -----"

echo ""
echo "installing emade..."
DATADIR=../../datasets/emade
mkdir -p $DATADIR
git config --global credential.helper cache
git clone https://github.gatech.edu/emade/emade.git --branch CacheV2 $DATADIR
echo "...emade done"

echo ""
echo "adding new dependencies...make sure you've activated ezcgp-py"
#conda activate ezcgp-py
conda install keras
conda install sqlalchemy
conda install dill
conda install scikit-image
pip install lxml
pip install deap
pip install lmfit
pip install hmmlearn
pip install sep
pip install spectral
# maybe make a new emade + ezcgp conda env
