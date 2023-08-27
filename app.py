%%writefile app.py
import streamlit as st
from threading import Thread
from typing import Iterator
import torch
import accelerate
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


model_id = 'codellama/CodeLlama-7b-Instruct-hf'
cache_dir='model'

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id,cache_dir=cache_dir)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        config=config,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map='auto',
        use_safetensors=False,
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful,
unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
\n\nIf a question does not make any sense, or is not factually coherent,
explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.\
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
        do_sample=True,
        max_new_tokens = 1024,
        temperature = 0.1,
        top_p = 0.9,
        top_k = 50,
        num_beams=1)
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)

st.title("Code Llama 7B Chat on Colab GPU by Umar IGAN")
st.markdown("""
    This Space demonstrates model [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) by Meta, a Code Llama model with 7B parameters fine-tuned for chat instructions and specialized on code tasks.
    I build this on colab in a way to use your colab gpu on this interface.
    """)

message = st.text_area("Type a message...")
system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT)
max_new_tokens = 1024,
temperature = 0.1,
top_p = 0.9,
top_k = 50,


if st.button("Submit"):
    history = []
    input_token_length = get_input_token_length(message, history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        st.error(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')
            return

        
    output_elem = st.empty()  # Create an empty element for dynamic text update
    def update_output(response):
        history.append((message, response))
        output_elem.text(response)  # Update the element's content with the response

    generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
    for response in generator:
        #st.write(response)  # Display response in the sidebar as well
        update_output(response)  # Update the dynamic text element with the response


if st.button("Clear"):
    st.text_area("Type a message...", value="")

if st.button("Retry"):
    pass  # Implement the retry logic here
