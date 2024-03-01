import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI

from agentlite.llm.LLMConfig import LLMConfig
from agentlite.llm.utils import get_response, post_http_request

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

FASTCHAT_MODELS = ["vicuna-7b", "zephyr-7b-beta", "lam-7b-v1", "lam-7b-v2"]

XGEN_NAMES = ["Salesforce/xgen-7b-4k-base", "Salesforce/xgen-7b-8k-base"]
INS_XGEN_NAMES = ["Salesforce/xgen-7b-8k-inst"]
VLLM_NAMES = ["sfr"]


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
        self.client = OpenAI(api_key=llm_config.openai_api_key)

    def run(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
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
            openai_api_key=llm_config.openai_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
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
            openai_api_key=llm_config.openai_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def run(self, prompt: str):
        return self.llm_chain.run(prompt)


class langchain_local_llm(LangchainLLM):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)
        openai.api_key = "EMPTY"  # Not support yet
        openai.base_url = "http://localhost:8000/v1"


class fast_llm(BaseLLM):
    # using fastchat llm server
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)

    def format_prompt(self, prompt: str, end_of_prompt: str) -> str:
        return prompt.strip() + " " + end_of_prompt

    def run(self, prompt: str) -> str:
        openai.api_key = "EMPTY"  # Not support yet
        openai.base_url = "http://localhost:8000/v1/"
        prompt = self.format_prompt(prompt, self.end_of_prompt)
        completion = openai.completions.create(
            model=self.llm_name,
            temperature=self.temperature,
            stop=self.stop,
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        output = completion.choices[0].text
        return output


class vllm_api_llm(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)
        self.api_url = ""
        self.n = 1
        self.stream = False
        self.trial = 0

    def run(self, prompt: str):
        # completion = openai.Completion.create(model=self.llm_name, temperature=temperature, stop=stop,
        #                                     prompt=prompt, max_tokens=128)
        done = False
        while not done:
            try:
                response = post_http_request(
                    prompt=prompt,
                    api_url=self.api_url,
                    n=self.n,
                    use_beam_search=False,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                output = get_response(response)
                done = True
                return output[0]
            except BaseException as ex:
                print(str(ex))
                self.trial += 1
                if self.trial > 5:
                    done = True
        return "No response"


def get_llm_backend(llm_config: LLMConfig):
    llm_name = llm_config.llm_name
    if llm_name in OPENAI_CHAT_MODELS:
        return LangchainChatModel(llm_config)
    elif llm_name in OPENAI_LLM_MODELS:
        return LangchainLLM(llm_config)
    elif llm_name in VLLM_NAMES:
        return vllm_api_llm(llm_config)
    else:
        return fast_llm(llm_config)
