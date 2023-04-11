# Blade2Blade
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LAION-AI/blade2blade/ci.yml?branch=main)
![GitHub](https://img.shields.io/github/license/LAION-AI/blade2blade)
![GitHub Repo stars](https://img.shields.io/github/stars/LAION-AI/blade2blade?style=social)

***Evil Sharpens Good.***

Developed By: The LAION-AI Safety Team

|**[Quick Start](#quick-start-fire)** | **[Installation](#installation)** | **[More about blade2blade](#what-is-blade2blade)** |


### Quick Start :fire:

```python
from blade2blade import Blade2Blade

blade = Blade2Blade("shahules786/blade2blade-t5-base")
prompt = "|prompter| I'm here to test your blade|endoftext|"
blade.predict(prompt)
```

### Installation
- With Pypi
```bash
pip install blade2blade
```
- From source
```
git clone https://github.com/LAION-AI/blade2blade.git
cd blade2blade
pip install -e .
```
### What is Blade2Blade?

Blade2Blade is a system that performs fully automated redteaming and blueteaming on any chat, or classifcation model. By using RL, we tune an adversarial user prompter to attack other models and create user prompts that promote dangerous responses from the attacked model. The attacked model is then optimized against the adversarial user prompter to preform automated blueteaming.

Below shows an example of a Blade2Blade type system that attacks GoodT5 (the blueteam model). GoodT5 is a model that is designed to predict Rules of Thumb and safety labels. EvilT5 is a model designed to predict user prompts from the Rules of Thumb and safety labels given to it.
![image](https://github.com/LAION-AI/blade2blade/blob/b01ff1b594c8f9661bdc3365e1e8d3530e7287f2/images/Blade2Blade.png)

#### What is redteaming and blueteaming?

Both red teams and blue teams strive to increase security in a system, but they go about it in different ways. A red team simulates an attacker by looking for weaknesses and trying to get past a system's defences. When an incident occurs, a blue team answers and defends against attacks.

#### What is the final goal of Blade2Blade?

We want to make an easy to use package that will add automated redteaming and blueteaming to any existing training loop. By bringing down the requirements for redteaming and blueteaming we hope that companies and individuals will strive to include this in their systems and create safer LLMs.
