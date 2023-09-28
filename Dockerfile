FROM gcr.io/deeplearning-platform-release/base-gpu.py310

RUN apt-get update

WORKDIR /root

#install sd libraries
RUN git clone -b v0.14.0 https://github.com/huggingface/diffusers.git
WORKDIR diffusers
RUN git checkout f20c8f5a1aba27f5972cad50516f18ba516e4d9e
WORKDIR /root
RUN pip install /root/diffusers

RUN git clone https://github.com/huggingface/peft.git
RUN pip install /root/peft 
RUN git clone https://huggingface.co/spaces/smangrul/peft-lora-sd-dreambooth

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt (model link)

#install libraries
# RUN pip install -U xformers safetensors tqdm ftfy loralib evaluate psutil pyyaml packaging bitsandbytes==0.35.0 datasets

RUN pip install xformers==0.0.18 \
    safetensors==0.3.0 \
    tqdm==4.65.0 \
    ftfy==6.1.1 \
    loralib==0.1.1 \
    evaluate==0.4.0 \
    psutil==5.9.4 \
    pyYAML==6.0 \
    packaging==21.3 \
    datasets==2.11.0 \
    bitsandbytes==0.35.0 \
    transformers==4.27.4 \
    accelerate==0.18.0 \
    Jinja2==3.1.2 \
    cloudml-hypertune==0.1.0.dev6 \
    google-cloud-aiplatform \
    google-cloud-storage

COPY keyfile.json .
COPY scripts/infer.py infer.py
COPY scripts/train_wo_nfs.py train_wo_nfs.py

# Set the Google Cloud Service Account Credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="keyfile.json"
# Set the environment variable for your service account key path (replace YOUR_SERVICE_ACCOUNT_KEY.json with the actual file name)
ENV SERVICE_ACCOUNT_KEY_PATH "keyfile.json"

# Copies the trainer code to the docker image.
# COPY train_wo_nfs.py /root/train_wo_nfs.py 

# Sets up the entry point to invoke the trainer.
# ENTRYPOINT ["python3", "-m", "train_wo_nfs"]

