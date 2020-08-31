FROM fastdotai/fastai:latest

RUN pip --no-cache-dir install deepflash2

COPY nbs/* deepflash2_notebooks/
