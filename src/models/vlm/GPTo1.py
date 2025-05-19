import sys
import os
import base64
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from openai import OpenAI
import json
from src.types.BaseModel import BaseModel
from typing import Dict, Any
import time

# os.environ["http_proxy"] = "http://localhost:7897"
# os.environ["https_proxy"] = "http://localhost:7897"
class GPTo1Model(BaseModel):
    def __init__(self, api_key):
        super().__init__(
            model_name="o1",
            base_url="",
            api_key=api_key
        )
        self.client = OpenAI(api_key=api_key)

    def get_image_code(self, image_path):
        with open(image_path, 'rb') as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        return base64_image

    def generate_answer(self, test_case: Dict, image_root_path: str):
        prompt = self.format_prompt(test_case)

        # Get image file paths based on the image indices
        # raw_image_indices = test_case['test_case']['input'].get('image_index', '')
        raw_image_indices = test_case['test_case']['input'].get('image_index', [])

        if isinstance(raw_image_indices, list):
            image_paths = [os.path.join(image_root_path, f'{img_id}.png') for img_id in raw_image_indices]
        else:
            image_paths = [os.path.join(image_root_path, f'{raw_image_indices}.png')]

        start_time = time.time()

        try:
            messages = [prompt["system"], prompt["user"]]

            for path in image_paths:
                image_code = self.get_image_code(path)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_code}"
                            }
                        }
                    ]
                })

            completion = self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )


            content = completion.choices[0].message.content

            usage = completion.usage

            predicted_answer = content
            token_count = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
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


