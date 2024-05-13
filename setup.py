from setuptools import find_packages, setup


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip()
            for line in file_content.strip().split("\n")
            if not line.startswith("#")
        ]
        return lines


setup(
    name="agentlite-llm",
    version="0.1.12",
    url="https://github.com/SalesforceAIResearch/AgentLite",
    description="Light Library for Building LLM Agent System",
    packages=find_packages(exclude=["test*", "app*", "doc*", "example"]),
    python_requires=">=3.9",
    install_requires=get_requires(),
    license="Apache License 2.0",
    author="Zhiwei Liu",
    author_email="zhiweiliu@salesforce.com",
)
