FROM python:3.5
run apt-get update
run apt-get install -y python3-pip wget
run pip3 install numpy scipy scikit-learn
add ./program/CRF++-0.58.tar.gz crf.tar.gz
run cd /crf.tar.gz/CRF++-0.58; ./configure; make; make install
env LD_LIBRARY_PATH=/usr/local/lib
workdir /code