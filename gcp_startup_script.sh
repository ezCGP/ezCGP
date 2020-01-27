# startup script for GCP instance to install Anaconda, packages, and ezCGP repo
# setup for multiple GPUs environment
sudo apt-get update
sudo apt-get install bzip2 libxml2-dev git python3-pip -y
# get the install.sh script from Github and put it in local directory for user to run manually
wget https://raw.githubusercontent.com/ezCGP/ezCGP/2020S-gpu/install.sh
wget https://raw.githubusercontent.com/ezCGP/ezCGP/2020S-gpu/requirements.txt
bash install.sh
