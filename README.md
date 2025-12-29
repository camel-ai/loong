<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="assets/banner.png" alt=banner>
  </a>
</div>

</br>

<div align="center">

[![Discord][discord-image]][discord-url]
[![X][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![Wechat][wechat-image]][wechat-url]
[![Wechat][loong-image]][loong-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Star][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]

<h4>

[Community](https://github.com/camel-ai/camel#community) |
[Cookbook](https://github.com/camel-ai/loong/tree/main/cookbooks) |
[Datasets](https://huggingface.co/datasets/camel-ai/loong) |
[Loong Blog](https://www.camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers) |
[Contributing](data/CONTRIBUTING.md) |
[CAMEL-AI](https://www.camel-ai.org/)
</h4>

<br>

🐉 Loong Project is a collaborative effort to explore whether reasoning-capable models can bootstrap themselves from small, high-quality **seed datasets** by generating synthetic data and verifying LLM agent responses.

<br>

🌟 Star **Loong** on GitHub to stay updated, or join our **Initiative Program**

<img src="assets/star.gif" width="40px">

We invite researchers and developers to contribute seed datasets, verifiers, and ideas to help improve and extend our project. Ready to join? Click the link below to apply now.

<br>
  
<a href="https://www.camel-ai.org/collaboration-questionnaire">
    <img src="assets/initiative_banner.png" alt=initiative_banner width="50%">
</a>

</div>

<br>

---

## Agent-Environment Loop

<div align="center">
  <img src="assets/loongdiagram.png" alt="Project Loong Agent-Environment Loop" width="100%">
</div>

Project Loong leverages a **Generator** to create synthetic questions/answers from seed datasets, while a **Verifier** evaluates 
the correctness of those responses. A **Trainable Agent** then learns iteratively from these verified Q&As, enabling scalable 
self-improvement through reinforcement learning and more advanced strategies.

---


## 🔍 What's in this Repo?

#### 1. 📊 **Seed Datasets** — Real, human-vetted data from computable domains like math, physics, finance, etc.
#### 2. 📘 **Cookbooks** — Modular scripts for synthetic data generation, verification, and RL training loops.

<br>

## 🚀 Getting Started

To set up the project locally, follow these steps:

```bash
# Create a virtual environment with Python 3.10
uv venv .venv --python=3.10

# Activate the virtual environment
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .
```

<br>

## 📊 Seed Datasets [→](./data)
A collection of **seed datasets**, structured for generation and verification, divded for each domain. 

Each datapoint includes:
- `question`
- `final_answer`
- `rationale` (typically code)
- `metadata` (license, source, domain, required_dependencies, name, contributor, date_created, difficulty, tags, anything else)

Each dataset is designed to allow automatic evaluation via **verifiers**, usually by executing the rationale code and comparing the output to the known answer.

###  Dataset Overview

The repository currently includes a total of **8,729 questions** spanning **12 diverse domains** (and growing!):

- 🧮 **Advanced Math:** 1,611 questions  
- ⚛️ **Advanced Physics:** 429 questions   
- 🧪 **Chemistry:** 3076 questions   
- 🧬 **Computational Biology:** 51 questions  
- 💹 **Finance:** 235 questions
- 🎮 **Game:** 926 questions
- 📈 **Graph & Discrete Math:** 178 questions  
- 🧠 **Logic:** 1,054 questions  
- 📐 **Mathematical Programming:** 76 questions
- 💊 **Medicine:** 916 questions
- 🔒 **Security & Safety:** 516 questions
- 🧑‍💻 **Programming:** 585 questions

You can find each domain's dataset in the corresponding folder. 

> [!TIP]
> Want to contribute your own? See the [CONTRIBUTING.md](data/CONTRIBUTING.md) for seed datasets.

<br>

## 📘 Cookbooks [→](./cookboos)
Reusable scripts and notebooks for:

- Few-shot prompting from seed data
- Generating synthetic questions, rationales, and answers
- Running verifiers over generations
- Exporting datasets for supervised fine-tuning or RL

These pipelines allow you to condition generations on real data, verify outputs, and build consistent synthetic traces.

<br>

## 🧬 Contributuing to Project Loong 🐉

We're looking for:
- Seed datasets in verifiable domains
- New verifiers
- Cookbook improvements
- Experimental environments for RL

<br>

>
> We greatly appreciate your interest in contributing to our open-source initiative. To ensure a smooth collaboration and the success of contributions, we adhere to a set of contributing guidelines similar to those established by CAMEL. For a comprehensive understanding of the steps involved in contributing to our project, please refer to the CAMEL [Contributing Guidelines](data/CONTRIBUTING.md). 🤝
>

<br>

## 📜 License

- Code: [LICENSE](./LICENSE)
- Data: Per-dataset license in `metadata.json`

<br>

## 👥 Maintainers & Contact

Project Loong is led by the [CAMEL](https://www.camel-ai.org/) team, with contributors from across the open-source AI research community.

If you're keen on exploring new research opportunities or discoveries with our platform and wish to dive deeper or suggest new features, we're here to talk. Feel free to get in touch for more details at camel-ai@eigent.ai.

<br>

- Join us ([*Discord*](https://discord.camel-ai.org/) or *WeChat*) in pushing the boundaries of finding the scaling laws of agents.

<div align="center">
  <img src="assets/wechat_qr.jpg" width="200" alt="WeChat QR Code">
</div>

## 🖊️ Cite

If you find this repo useful, please cite:


```
@misc{loong2025,
  title={Loong: Synthesize Long CoTs at Scale through Verifiers},
  author={CAMEL-AI.org},
  url={https://github.com/camel-ai/loong},
  year={2025}
}
```

[discord-image]: https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb
[discord-url]: https://discord.camel-ai.org/
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[huggingface-url]: https://huggingface.co/datasets/camel-ai/loong
[loong-image]: https://img.shields.io/badge/WeChat-ProjectLoong-brightgreen?logo=wechat&logoColor=white
[loong-url]: ./assets/wechatgroup.jpeg
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/loong/blob/main/LICENSE
[reddit-image]: https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white
[reddit-url]: https://www.reddit.com/r/CamelAI/
[star-image]: https://img.shields.io/github/stars/camel-ai/loong?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/loong/stargazers
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[wechat-url]: ./assets/wechat_qr.jpg
[x-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social
[x-url]: https://x.com/CamelAIOrg
