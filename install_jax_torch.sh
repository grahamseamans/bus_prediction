PYTHON_VERSION=cp38  
CUDA_VERSION=cuda101  # alternatives: cuda100, cuda101, cuda102, cuda110, check your cuda version
PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
JAX_VERSION=0.1.70
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-$JAX_VERSION+$CUDA_VERSION-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade jax 

BASE_URL='https://download.pytorch.org/whl'
CUDA_VERSION=cu101
TORCH_VERSION=1.8.1
pip install --upgrade $BASE_URL/$CUDA_VERSION/torch-$TORCH_VERSION%2B$CUDA_VERSION-$PYTHON_VERSION-$PYTHON_VERSION-linux_x86_64.whl