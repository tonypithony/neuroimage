# source TORCHTEST/bin/activate

# pip install --upgrade pip
# pip install ollama transformers
# pip install --upgrade diffusers[torch]

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import ollama

original_image = 'i.png'

response = ollama.chat(
    model='gemma3:4b',
    messages=[{
        'role': 'user',
        'content': 'What is in this image? Describe in the details',
        'images': [original_image]
    }],
    options={
             'temperature': 0.4, # значение от 0,0 до 0,9 (или 1) определяет уровень креативности модели или ее неожиданных ответов.
              #'top_p': 0.9, #  от 0,1 до 0,9 определяет, какой набор токенов выбрать, исходя из их совокупной вероятности.
              #'top_k': 90, # от 1 до 100 определяет, из скольких лексем (например, слов в предложении) модель должна выбрать, чтобы выдать ответ.
              #'num_ctx': 500_000, # устанавливает максимальное используемое контекстное окно, которое является своего рода областью внимания модели.
             'num_predict': 250, # задает максимальное количество генерируемых токенов в ответах для рассмотрения (100 tokens ~ 75 words).
    		}
)

# print(response.message.content)

repo_id = "stabilityai/stable-diffusion-2-base"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = response.message.content
print(f"Полученный запрос: {prompt}")
image = pipe(prompt, num_inference_steps=150).images[0]
image.save(f'neuro{original_image[:-4]}.png')