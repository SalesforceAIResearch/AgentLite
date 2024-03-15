nohup python evaluate_webshop.py --llm xlam_v2 --agent_arch react > nohup.out 2>&1 &
nohup python evaluate_webshop.py --llm xlam_v2 --agent_arch act > nohup.out 2>&1 &
nohup python evaluate_webshop.py --llm xlam_v2 --agent_arch bolaa > nohup.out 2>&1 &
nohup python evaluate_webshop.py --llm gpt-4-0613 --agent_arch bolaa > nohup.out 2>&1 &
nohup python evaluate_webshop.py --llm gpt-3.5-turbo-16k --agent_arch bolaa > nohup.out 2>&1 &
nohup python evaluate_webshop.py --llm gpt-3.5-turbo-16k --agent_arch react > nohup.out 2>&1 &