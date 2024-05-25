# Benchmark Evaluation Instructions
## HotpotQA
We follow [BOLAA](https://github.com/salesforce/BOLAA) environment to design the hotpotqa benchmark. And we designed the invidual agent with AgentLite. Run evaluation with 
```
pip install joblib
cd hotpotqa
python evaluate_hotpot_qa.py --llm gpt-4-0613 --agent_arch act
```

## Webshop
We follow [AgentBoard](https://github.com/hkust-nlp/AgentBoard) environment to setup the webshop benchmark. And we designed the individual agent via AgentLite with Search and Click actions.
1. Follow [setup](https://github.com/hkust-nlp/AgentBoard#setup-environment) in the AgentBoard to run the webshop in backend. Go to your home directory first and running the following scripts from AgentBoard.  
```
conda create -n agentboard python=3.8.13  # python version should be 3.8.13
conda activate agentboard
git clone https://github.com/hkust-nlp/AgentBoard.git
cd AgentBoard
mkdir data
wget https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz
tar -zxvf data.tar.gz
INSTALL_WEBARENA=false bash ./setup.sh
cd ./agentboard/environment/WebShop
bash ./run_dev.sh
```
it is highly suggested running webshop in backend with `tmux`.

2. Since AgentLite is using a different python version, you should create a new environment for AgentLite.
3. Run AgentLite evaluation in this folder  with
```
cd webshop
python evaluate_webshop.py --llm gpt-4-0613 --agent_arch act
```

## Tool-query
We follow [AgentBoard](https://github.com/hkust-nlp/AgentBoard) environment to setup the tool-query benchmark. And we designed the individual agent via AgentLite with all the corresponding function call as actions.
You should first get a `data/tool-query` folder, which is a copy of data from AgentBoard with
```shell
cp -rf [AgentBoardPath]/data/tool-query ./tool/
```
Then, you could directly run evaluation in this folder with
```
cd tool
python evaluate_tools.py --llm gpt-4-0613 --agent_arch react
```

## Tool-operation
We follow [AgentBoard](https://github.com/hkust-nlp/AgentBoard) environment to setup the tool-operation benchmark. And we designed the individual agent via AgentLite with all the corresponding function call as actions.
You should first get a `data/tool-operation` folder, which is a copy of data from AgentBoard with
```shell
cp -rf [AgentBoardPath]/data/tool-operation ./tool-operation/data/
```
Then, you could directly run evaluation in this folder with
```
cd tool
python evaluate_tool_operation.py --llm gpt-4-32k --agent_arch react
```



## Agent Architectures
You can substitute the `--agent_arch` with different architectures as in [BOLAA](https://github.com/salesforce/BOLAA), including `react`, `act`, `planact`, `planreact`, `zs`, `zst`, `bolaa`. The multi-agent architecture of hotpotqa will soon be released.

Note that the `bolaa` implementation is slightly different from the [original paper](https://arxiv.org/abs/2308.05960) due to the communication template is different in AgentLite implementation. You could change it in your best practice. 


## Local LLM Inference
We suggest using url-based way for inference, such as [fastchat](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md) to get response from local model. 
We provides the [example code lines](https://github.com/SalesforceAIResearch/AgentLite/blob/3b40821ab3c6358947205ede1ed933f906f219e9/benchmark/webshop/evaluate_webshop.py#L23-L31) in benchmark evaluation code. 
You only need to change the `base_url`, `llm_name` for your local model inference.
