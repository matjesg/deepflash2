# FROM pytorch/pytorch
FROM pytorch/pytorch

RUN apt-get update && apt-get install -y software-properties-common rsync ffmpeg libsm6 libxext6
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz && apt-get update
RUN pip --no-cache-dir install opencv-python \
    git+https://github.com/MouseLand/cellpose.git@316927eff7ad2201391957909a2114c68baee309 \
    SimpleITK>=2.0.2 \
    kornia \
    tensorboard \
    wandb

RUN pip --no-cache-dir install deepflash2
RUN echo '#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser' >> run_jupyter.sh
RUN chmod u+x run_jupyter.sh

COPY nbs/* nbs/
COPY paper/* paper/
COPY deepflash2_GUI.ipynb ./


