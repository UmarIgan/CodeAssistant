%%writefile app.py
import streamlit as st
from threading import Thread
from typing import Iterator
import torch
import accelerate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


model_id = 'codellama/CodeLlama-7b-Instruct-hf'

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map='auto',
        use_safetensors=False,
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000



def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)

def main():
    st.title("Code Llama 13B Chat")
    st.markdown("""
    This Space demonstrates model [CodeLlama-13b-Instruct](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) by Meta, a Code Llama model with 13B parameters fine-tuned for chat instructions and specialized on code tasks.
    """)

    message = st.text_area("Type a message...")
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT)
    max_new_tokens = st.slider("Max new tokens", min_value=1, max_value=4096, value=1024)
    temperature = st.slider("Temperature", min_value=0.1, max_value=4.0, value=0.1, step=0.1)
    top_p = st.slider("Top-p (nucleus sampling)", min_value=0.05, max_value=1.0, value=0.9, step=0.05)
    top_k = st.slider("Top-k", min_value=1, max_value=1000, value=50)

    if st.button("Submit"):
        history = []
        generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
        for response in generator:
            history.append((message, response))
            st.write(response)

    if st.button("Clear"):
        st.text_area("Type a message...", value="")

    if st.button("Retry"):
        pass  # Implement the retry logic here

if __name__ == "__main__":
    main()
