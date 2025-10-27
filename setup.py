from setuptools import setup, find_packages

setup(
    name="COG", 
    version="0.1.0",
    author="Yingzhi Mao, Chunkang Zhang, Junxiang Wang, Xinyan Guan, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun",
    author_email="maoyingzhi2024@iscas.ac.cn",
    description="A training framework for Large Reasoning Models that enhances safety while preserving reasoning ability.",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "datasets>=2.0",
        "accelerate>=0.25",
        "tqdm>=4.60",
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "sentencepiece>=0.2.0",
        "peft>=0.15.2",
        "trl>=0.9.6",
        "vllm>=0.9.1",
        "xformers>=0.0.30",
        "gradio>=5.31.0",
        "openai>=1.93.0",
    ],
    python_requires=">=3.10",
)
