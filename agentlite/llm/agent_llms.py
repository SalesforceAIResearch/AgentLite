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
        llm = OpenAI(model_name = self.llm_name,
                     openai_api_key = llm_config.openai_api_key, 
                     temperature = self.temperature,
                     max_tokens = self.max_tokens
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
        llm = ChatOpenAI(model_name = self.llm_name,
                     openai_api_key = llm_config.openai_api_key, 
                     temperature = self.temperature,
                     max_tokens = self.max_tokens
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
    
    def format_prompt(self, prompt: str, end_of_prompt: str)-> str:
        return prompt.strip() + " " + end_of_prompt
    
    def run(self, prompt: str)-> str:
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


if __name__ == "__main__":
    # prompt = "The world is"
    # llm_name = "Salesforce/xgen-7b-4k-base"
    # llm_name = "mpt-30b-instruct"
    # llm_name = "longchat-7b-16k"
    # llm_name = "gpt-3.5-turbo"
    # llm = vicuna_llm
    # llm_name = "vicuna-7b"
    # llm = fast_llm(llm_name).run
    llm_name = "sfr"
    llm_config_dict = {"llm_name": llm_name, "temperature": 0.0, "context_len": 4096}
    llm_config = LLMConfig(llm_config_dict)
    llm = get_llm_backend(llm_config).run
    # llm = langchain_llm(llm_name).run
    # llm = openai_chatllm(llm_name).run
    # llm = longchat_llm
    one_shot = """
Instruction:
Find me machine wash men's t-shirts with long sleeve with color: country, and size: xx-large,
and price lower than 30.00 dollars
Action:
search[men's t-shirts with long sleeve]


Observation:
[ Back to Search ]
Page 1 (Total results: 50)
[ Next > ]

[ B09Q67H373 ]
JSPOYOU 2022 Newly T-Shirt for Mens Funny 3D Graphics Pattern Crewneck Short Sleeve Tees Big
and Tall Summer Casual Comfy Top
$1.99 to $8.99

[ B09S3BN15C ]
Mens Linen Shirt,Men's Striped Shirts Casual Short Sleeve Button Down Shirts Regular Fit
Hawaiian Shirts Beach Tees Tops
$3.78 to $11.38

[ B09Q8RD8YN ]
Bungo Stray Anime Dogs Anime Character, Long Sleeve, Sweatshirt, Hoodie, T shirt
$19.99

[ B09KLQLLT2 ]
Long Sleeve Superhero T Shirt Tank Top Mens Compression Shirt Men Workout Fitness Gym Shirt
$19.99

[ B00O30JLDK ]
Legendary Whitetails Men's Non-Typical Long Sleeve T-Shirt
$10.52 to $40.5

[ B09HGFV91W ]
ZHDD Tops for Mens, Men's Knight Punk Vintage Style Long Sleeve Asymmetric Hem Cowl Neck Teen
Boys Gothic Pullover
$4.98 to $7.99
Action:
think[I am simulating the action of clicking on that product to get more information about it such as the available colors, sizes, and the exact price. Based on the results, I should select the item [B00O30JLDK] as it seems to fit your criteria: a long sleeve men's t-shirt that might be under $30. However, I need to verify these details including the color, size, and final price. After the "click", I'll be able to provide more detailed information on this particular product. Therefore, my next action should be click[B00O30JLDK].]

Observation:
OK
Action:
click[B00O30JLDK]


Observation:
[ Back to Search ]
[ < Prev ]
color
  [ army ]
  [ big game field camo ]
  [ black ]
  [ cardinal ]
  [ charcoal heather ]
  [ inferno ]
  [ mossy oak bottomland ]
  [ mossy oak country ]
  [ mossy oak country dna ]
  [ mossy oak coyote ]
  [ slate ]
  [ swamp ]
  [ turkey ]
  [ flag ]
  [ country ]
  [ fish ]
size
  [ small ]
  [ medium ]
  [ large ]
  [ large tall ]
  [ x-large ]
  [ x-large tall ]
  [ xx-large ]
  [ xx-large big tall ]
  [ 3x-large big ]
  [ 3x-large big tall ]
  [ 4x-large big ]
  [ 4x-large big tall ]
  [ 5x-large big ]
Legendary Whitetails Men's Non-Typical Long Sleeve T-Shirt
Price: $10.52 to $40.5
Rating: N.A.
[ Description ]
[ Features ]
[ Reviews ]
[ Buy Now ]


Action:
think[The "Legendary Whitetails Men's Non-Typical Long Sleeve T-Shirt" is available in a variety of colors, including the requested color "country". It is also available in the requested size, "xx-large". However, the price range for this item is from $10.52 to $40.5. This suggests that depending on the specific color and size selection, the price could potentially exceed the specified budget of $30.]
Observation:
OK


Action:
click[country]
Observation:
You have clicked country.


Action:
click[xx-large]
Observation:
You have clicked xx-large.


Action:
click[Buy Now]
"""

    instruction = "i want to find decorative, multi-colored vinyl dots for my living room windows. the size should be 17.7 inches by 23.6 inches, and price lower than 90.00 dollars"
    prompt = f"{one_shot}\n\nInstruction:\n{instruction}\n\nActions:\n"
    a = llm(prompt, 0.8, ["\n"])
    # a = llm(prompt)
    print(a)
