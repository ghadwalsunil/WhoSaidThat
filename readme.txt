python3.8 -m venv venv
source /home/sunil/projects/Stuff/Combined/venv/bin/activate
pip install --upgrade pip

git clone https://github.com/TaoRuijie/TalkNet-ASD.git
cd TalkNet-ASD
pip install -r requirement.txt

pip3 install jupyter
pip install notebook==6.4.4
pip install traitlets==5.9.0

pip install -qq pyannote.audio
pip install -qq rich

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip install matplotlib
pip install python-dotenv

pip install face-recognition
pip install scikit-learn
pip install openpyxl