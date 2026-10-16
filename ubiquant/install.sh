#!/bin/bash

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
pip install --no-cache-dir "vllm==0.11.0"

echo "2. install basic packages"
pip install "transformers[hf_xet]==4.57.3" "accelerate==1.11.0" "datasets==4.0.0" "peft==0.17.1" hf-transfer \
    "numpy==1.26.4" "pyarrow>=15.0.0" "pandas==2.3.3" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "torchdata==0.11.0" \
    "ray[default]==2.53.0" codetiming "hydra-core==1.3.2" pylatexenc qwen-vl-utils "wandb==0.24.2" dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"

# Install flash-attn-2.8.0.post2
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout de79b1361e0e6ade09b9b5ca3949cc088eee9965
python setup.py install

pip install --no-cache-dir flashinfer-python==0.3.1


echo "4. install TransformerEngine and Megatron"
echo "Notice that TransformerEngine installation can take very long time, please be patient"
pip install "onnxscript==0.3.1"
pip install nvidia-mathdx==25.6.0

MAX_JOBS=128 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

#export NVTE_FRAMEWORK=pytorch && MAX_JOBS=128 NVTE_BUILD_THREADS_PER_JOB=4 pip3 install --resume-retries 999 --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.8

#pip install --no-deps --no-cache-dir git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.14.0rc7

#pip install -U git+https://github.com/ISEEKYAN/mbridge.git

pip install nvidia-cudnn-cu12==9.10.2.21

pip install -r requirements.txt

pip install math-verify==0.8.0

echo "Successfully installed all packages"
