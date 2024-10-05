from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI
from huggingface_hub import model_info
from huggingface_hub.utils._errors import RepositoryNotFoundError

from agentlite.llm.LLMConfig import LLMConfig

OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
]
OPENAI_LLM_MODELS = ["text-davinci-003", "text-ada-001"]

def is_huggingface_model(llm_name: str) -> bool:
    """Check if the model is available on the Hugging Face Hub"""
    try:
        model_info(llm_name)
        return True
    except RepositoryNotFoundError:
        return False


class BaseLLM:
    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_name = llm_config.llm_name
        self.context_len: int = llm_config.context_len
        self.stop: list = llm_config.stop
        self.max_tokens: int = llm_config.max_tokens
        self.temperature: float = llm_config.temperature
        self.end_of_prompt: str = llm_config.end_of_prompt

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)

    def run(self, prompt: str):
        # return str
        raise NotImplementedError


class OpenAIChatLLM(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)
        self.client = OpenAI(api_key=llm_config.api_key)

    def run(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

class VllmChatModel(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    def run(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class LangchainLLM(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        from langchain_openai import OpenAI

        super().__init__(llm_config)
        llm = OpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def run(self, prompt: str):
        return self.llm_chain.run(prompt)


class LangchainChatModel(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        from langchain_openai import ChatOpenAI

        super().__init__(llm_config)
        llm = ChatOpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def run(self, prompt: str):
        return self.llm_chain.run(prompt)


# class LangchainOllamaLLM(BaseLLM):
#     def __init__(self, llm_config: LLMConfig):
#         from langchain_community.llms import Ollama

#         super().__init__(llm_config)
#         llm = Ollama(
#             model=self.llm_name,
#             temperature=self.temperature,
#             num_predict=self.max_tokens,
#             base_url=llm_config.base_url
#             # api_key=llm_config.api_key,
#         )
#         human_template = "{prompt}"
#         prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
#         self.llm_chain = LLMChain(prompt=prompt, llm=llm)

#     def run(self, prompt: str):
#         return self.llm_chain.run(prompt)

def get_llm_backend(llm_config: LLMConfig):
    llm_name = llm_config.llm_name
    llm_provider = llm_config.provider
    if llm_name in OPENAI_CHAT_MODELS:
        return LangchainChatModel(llm_config)
    elif llm_name in OPENAI_LLM_MODELS:
        return LangchainLLM(llm_config)
    elif is_huggingface_model(llm_name):
        return VllmChatModel(llm_config)
    else:
        return LangchainLLM(llm_config)
    # TODO: add more llm providers and inference APIs but for now we are using langchainLLM as the default
    # Using other LLM providers will require additional setup and configuration
    # We suggest subclass BaseLLM and implement the run method for the specific provider in your own best practices
