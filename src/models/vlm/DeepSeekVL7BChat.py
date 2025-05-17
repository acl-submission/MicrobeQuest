import os
import time
from pprint import pprint
from typing import Dict

import torch
from transformers import AutoModelForCausalLM
from src.models.vlm.deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from src.models.vlm.deepseek_vl.utils.io import load_pil_images
from src.types.BaseModel import BaseModel

class DeepSeekVLModel(BaseModel):
    def __init__(self, local_model_path):
        super().__init__(
            model_name="DeepseekVL",
            base_url="",
            api_key=""
        )

        # specify the path to the model
        self.use_model_name = "deepseek-ai/deepseek-vl-7b-chat"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.use_model_name)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(self.use_model_name, trust_remote_code=True, cache_dir=local_model_path)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    @staticmethod
    def get_image_paths(image_root_path, name):
        # Define the supported image file extensions
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

        # Used for storing the path of the found picture
        image_paths = []

        for root, _, files in os.walk(image_root_path):
            for file in files:
                # Check whether the file extension is in image format
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in IMAGE_EXTENSIONS:
                    # Check whether the file name contains "M001".
                    if name in file:
                        image_paths.append(os.path.join(root, file))
        return image_paths
    def generate_answer(self, test_case: Dict, image_root_path: str = ""):
        name = test_case.get("case_id").split("-")[1]
        image_paths = self.get_image_paths(image_root_path, name)
        content=""
        for _ in image_paths:
            content+="<image_placeholder>"
        prompt = self.format_prompt(test_case)
        start_time = time.time()
        messages = [prompt["system"], prompt["user"]]

        # print(test_case)
        messages.append({
            "role": "user",
            "content": test_case["test_case"]["input"],
        })

        try:
            # Model structuring
            conversation = [
                {
                    "role": "User",
                    "content": content + "Summarize the content of the above pictures." + f"{messages}",
                    "images": image_paths
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            predicted_answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            token_count = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        except Exception as e:
            print(f"[ERROR] Failed to call model: {e}")
            predicted_answer = "error"
            token_count = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        response_time = time.time() - start_time
        return self.format_result(
            test_case=test_case,
            predicted_answer=predicted_answer,
            prompt=prompt,
            response_time=response_time,
            token_count=token_count
        )
