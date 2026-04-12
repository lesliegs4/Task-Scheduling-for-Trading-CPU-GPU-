# 1. pull from image
FROM ghcr.io/nautechsystems/jupyterlab:nightly

# 2. set workdir
WORKDIR /home/nautilus

# 3. Install dependency to get data from github
RUN pip install "fsspec[http]"


