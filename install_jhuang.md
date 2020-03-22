```bash
sudo pip install virtualenv virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
mkdir -p ${WORKON_HOME}
# Add "source /usr/local/bin/virtualenvwrapper.sh" to ~/.bashrc
source ~/.bashrc
mkvirtualenv nuscenes --python=python3
# clone nuscenes-devkit
cd ~/repo/nuscenes-devkit
workon nuscenes
python3 -m pip install -r setup/requirements.txt
```
