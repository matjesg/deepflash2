# FROM pytorch/pytorch
FROM pytorch/pytorch

RUN apt-get update && apt-get install -y software-properties-common rsync ffmpeg libsm6 libxext6
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz
    
RUN pip --no-cache-dir install deepflash2 \
    SimpleITK \
    kornia \
    roifile

RUN pip --no-cache-dir install --no-deps \
    git+https://github.com/MouseLand/cellpose.git@316927eff7ad2201391957909a2114c68baee309

RUN ls
RUN git clone https://github.com/matjesg/deepflash2.git --depth 1 .

RUN echo '#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser' >> run_jupyter.sh
RUN chmod u+x run_jupyter.sh