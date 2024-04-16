import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to 🤖AgentLite🤖")
st.write("#### Lightweight Library for Building and Advancing Task-Oriented LLM Agent System")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    AgentLite is a research-oriented library designed for building and advancing LLM-based task-oriented agent systems. It simplifies the implementation of new agent/multi-agent architectures, enabling easy orchestration of multiple agents through a manager agent. Whether you're building individual agents or complex multi-agent systems, AgentLite provides a straightforward and lightweight foundation for your research and development. Check more details in [our paper](https://arxiv.org/abs/2402.15538).

    ## 🎉 News
    - **[03.2024]** [xLAM model](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4) and [xLAM code](https://github.com/SalesforceAIResearch/xLAM) is released! Try it with [AgentLite benchmark](./benchmark/), which is comparable to GPT-4!
    - **[03.2024]** We developed all the agent architectures in [BOLAA](https://arxiv.org/pdf/2308.05960.pdf) with AgentLite. Check our [new benchmark](./benchmark/)
    - **[02.2024]** Initial Release of AgentLite library and [paper](https://arxiv.org/abs/2402.15538)!

    ## 🌟 Key Features

    - **Lightweight Codebase**: Designed for easy implementation of new Agent/Multi-Agent architectures.
    - **Task-oriented LLM-based Agents**: Focus on building agents for specific tasks, enhancing their performance and capabilities.
    - **Research-oriented Design**: A perfect tool for exploring advanced concepts in LLM-based multi-agent systems.
    """
)