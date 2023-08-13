pip install git+https://github.com/cocodataset/panopticapi.git
pip install pytorch-fid
pip install fairscale==0.4.0
pip install fvcore termcolor cloudpickle tabulate black packaging
pip install yacs==0.1.8
pip install dataclasses
pip install iopath==0.1.7
pip install hydra-core==1.1
pip uninstall JWT
pip uninstall PyJWT
pip install PyJWT
# cd evaluate
git clone https://github.com/facebookresearch/detectron2
python -m pip install -e detectron2
# cd ..
# echo "detectron2 ok"