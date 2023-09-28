# Project Title

This project is a deep learning application that uses a Docker container to fine-tune a model on images using a technique called Dreambooth. The base model can be Stable Diffuser or any other model. The application is built on a base image from Google, which includes basic tools and libraries for deep learning, and CUDA drivers. The application also clones Diffuser's library and Dreambooth's library from GitHub.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker
- Google Cloud Platform account
- Google Cloud SDK

### Installing

Clone the repository to your local machine

Build the Docker image:

```bash
docker build -t your-image-name .
```

### Running the Application

The application consists of two scripts: a training script and an inference script. The training script fine-tunes a model on images, and the inference script generates predictions from the trained model.

To run the training script:

```bash
python3 train_wo_nfs.py --method=diffuser_dreambooth --model_name=Lykon/DreamShaper --input_storage=/gcs/envison-test/jobName/input --output_storage=/gcs/envison-test/jobName/model --prompt='photo of zwx person' --class_prompt='photo of person' --num_class_images=1 --lr=1e-6 --use_8bit=True --max_train_steps=1200 --text_encoder=True --set_grads_to_none=True
```

To run the inference script:

```bash
python3 infer.py --model_path=/gcs/envison-test/jobName/model --save_path=/gcs/envison-test/jobName/output/image
```

## Deployment

The application can be deployed on Google Cloud Platform using the AI Platform Custom Container Training Job. The job runs the training script and the inference script in sequence.

```py
# Define the necessary arguments and parameters for the job
args1 = [
    "--method=diffuser_dreambooth",
    "--model_name=Lykon/DreamShaper",
    f"--input_storage=/gcs/envison-test/testv1/input",
    f"--output_storage=/gcs/envison-test/{job_name}/model",
    "--prompt='photo of zwx person'",
    "--class_prompt='photo of person'",
    "--num_class_images=1",
    "--lr=1e-6",
    "--use_8bit=True",
    "--max_train_steps=1200",
    "--text_encoder=True",
    "--set_grads_to_none=True"
]
command1 = ["python3", "train_wo_nfs.py"] + args1

args2 = [
        f"--model_path=/gcs/envison-test/{job_name}/model",
        f"--save_path=/gcs/envison-test/{job_name}/output/image"
]
command2 = ["python3", "infer.py"] + args2

chained_command = f"{cmd1} && {cmd2}"
final_command = ["bash", "-c", chained_command]

# Create the job using aiplatform.CustomContainerTrainingJob
job = aiplatform.CustomContainerTrainingJob(
    display_name=job_name,
    container_uri=TRAINING_IMAGE_ENDPOINT,
    command=final_command,
    location=REGION,
    staging_bucket=STAGING_BUCKET
)

# Run the job
job.run(
    replica_count=1,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=1
)
```

## Built With

- Docker
- Google Cloud Platform
- Python

## Authors

- Umer Naeem

## Acknowledgments

- Hugging Face for the Diffuser and Dreambooth libraries
- Google for the base image and the AI Platform
