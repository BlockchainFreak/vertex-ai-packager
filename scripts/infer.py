import torch
import argparse
from diffusers import DiffusionPipeline, DDIMScheduler

default_negative_prompt = "amputee, bad anatomy, bad illustration, bad proportions, beyond the borders, blurry, body out of frame, boring background, branding, cropped, cut off, deformed, disfigured, dismembered, disproportioned, distorted, draft, duplicate, duplicated features, extra arms, extra fingers, extra hands, extra legs, extra limbs, fault, flaw, fused fingers, grains, grainy, gross proportions, improper scale, incorrect physiology, incorrect ratio, indistinct, logo, long neck, low quality, low resolution, animation, multiple people, extra bodies, extra face, multiple faces, multiple frames, broken teeth"

default_prompt_list = [
       "RAW photograph, centered and realistic. With fantasy elements, closeup of a handsome, white male, captivating, zwx person sitting in the glow of cinema lighting, as fire swirl and haze around him, creating a magical and dreamlike ambiance. Proper lighting. closeup of a handsome face looking at the viewer.",
    "RAW photograph, hyper realistic closeup image  of zwx person as a handsome character from Disney, dressed as a charming prince, prince charming, 4k, high quality. Extremely detailed facial closeup. Closeup image. fantasy backdrop. Perfect highlights and shadows on the face.",
    "RAW photograph, rough style A closeup shot of a resilient zwx person man, wearing rough clothes, in a desolate post-apocalyptic world filled with crumbling cityscapes and overgrown nature. . bleak, post-apocalyptic, somber, dramatic, highly detailed. Handsome face closeup",
    "RAW photograph, highly realisric, face closeup of zwx person as a stunning heavenly handsome man, sharp contrast, Techno Marble, photorealistic, creepy, art by gaston bussiere, hyper detailed, unreal engine 5, space, breathtaking, Korean light novel, photorealistic landscape, black background, cinematic still, handsome face closeup, green hues. ",
    "hyper realistic face closeup of zwx person as a Space Pirate, single man, Jolly roger Hat, Steampunk ship, cinematic Composition, Dramatic lighting , High Detail, 16k, UHD, HDR,",
    "hyper realistic closeup photo of a zwx person as a 1800s male assassin, old background, highly detailed. Extremely detailed facial closeup",
    "hyper realistic, frame-centered, closeup photo of a zwx person in the neonpunk setting. neon buildings around him. The person should be luminated perfectly. Neonpunk style, neon elemtns. Highly detailed closeup.",
    "hyper realistic image of zwx person, clean shave, wearing a  santa clause hat, snowy background, 8k, high detailed face closeup",
]

def main(args):
    # PROMPT = args.prompt
    MODEL_PATH = args.model_path
    SAVE_PATH = args.save_path
    PROMPT_LIST = args.prompts.split(";")
    NEGATIVE_PROMPT = args.negative_prompt
    
    # model_base = "runwayml/stable-diffusion-v1-5"

    pipe = DiffusionPipeline.from_pretrained(MODEL_PATH, safety_checker=None, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.unet.load_attn_procs(MODEL_PATH)
    pipe.to("cuda")
    
    generations = []
    for i, prompt in enumerate(PROMPT_LIST):
        print(f"Inference Batch: {i+1}/{len(PROMPT_LIST)}")
        images = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=512,
            width=512,
            num_inference_steps=40,
            num_images_per_prompt=4,
            guidance_scale=8,
        ).images
        generations += images

    for i, image in enumerate(generations):
        image.save(f"{SAVE_PATH}-{i+1}.png")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/path/to/model_weights", help="path to lora weights")
    parser.add_argument("--save_path", type=str, default="/result_images", help="path to save inference results")
    parser.add_argument("--prompts", type=str, default=";".join(default_prompt_list), help="default_prompt_list (separated by ';')")
    parser.add_argument("--negative_prompt", type=str, default=default_negative_prompt, help="negative prompt here")
                            
    args = parser.parse_args()
    print(args)
    main(args)