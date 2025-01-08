import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_MAX_NEW_TOKENS = 3000
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))


class MyModel():
    def __init__(self, model_id = "Llama-3.2-3B-Instruct") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        with open("questions.txt") as f:
            self.questions = f.readlines()

    def generate(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        max_new_tokens: int = 3000,
        temperature: float = 0.8,
        top_p: float = 0.75,
        top_k: int = 50,
        repetition_penalty: float = 1,
    ):
        conversation = []
        message = f"""
        You are an expert in various domains, and your goal is to provide detailed, accurate, and well-supported answers to questions. 
        When responding to queries, please follow these guidelines:
        1. Provide fact-based answers, drawing on reliable sources and logical reasoning.
        2. Present your response clearly and concisely, ensuring the user can easily understand.
        3. Focus on practical, actionable insights, offering valuable advice wherever applicable.
        4. Avoid uncertainty or ambiguity—answer with confidence and precision.
        
        Query: {message}
        """
        for user, assistant in chat_history:
            conversation.extend(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            )
        conversation.append({"role": "system", "content": message})

        input_ids = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(self.device)

        generate_kwargs = dict(
            {"input_ids": input_ids},
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )
        outputs = self.model.generate(**generate_kwargs)
        response =  self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response
