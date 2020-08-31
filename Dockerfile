FROM fastdotai/fastai:latest

RUN pip --no-cache-dir install deepflash2

COPY nbs/* deepflash2_notebooks/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]