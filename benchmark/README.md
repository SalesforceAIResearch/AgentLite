# Benchmark Evaluation Instructions
## HotpotQA
We follow [BOLAA](https://github.com/salesforce/BOLAA) environment to design the hotpotqa benchmark. And we designed the invidual agent with AgentLite. Run evaluation with 
```
cd hotpotqa
python evaluate_hotpot_qa.py
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

3. Since AgentLite is using a different python version, you should create a new environment for AgentLite.
2. Run AgentLite evaluation in this folder  with
```
cd webshop
python evaluate_webshop.py
```