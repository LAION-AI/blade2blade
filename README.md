# Blade2Blade
***Evil Sharpens Good.***

Developed By: The LAION-AI Safety Team

### What is Blade2Blade?

Blade2Blade is a system that performs fully automated redteaming and blueteaming on any chat, or classifcation model. By using RL, we tune an adversarial user prompter to attack other models and create user prompts that promote dangerous responses from the attacked model. The attacked model is then optimized against the adversarial user prompter to preform automated blueteaming.

Below shows an example of a Blade2Blade type system that attacks GoodT5 (the blueteam model). GoodT5 is a model that is designed to predict Rules of Thumb and safety labels. EvilT5 is a model designed to predict user prompts from the Rules of Thumb and safety labels given to it. All the data used in the training of these two models can be found in the prosocial dialogue dataset (https://huggingface.co/datasets/allenai/prosocial-dialog).
![image](https://github.com/LAION-AI/blade2blade/blob/b01ff1b594c8f9661bdc3365e1e8d3530e7287f2/images/Blade2Blade.png)

### What is redteaming and blueteaming?

Both red teams and blue teams strive to increase security in a system, but they go about it in different ways. A red team simulates an attacker by looking for weaknesses and trying to get past a system's defences. When an incident occurs, a blue team answers and defends against attacks.

### What is the final goal of Blade2Blade?

We want to make an easy to use package that will add automated redteaming and blueteaming to any existing training loop. By bringing down the requirements for redteaming and blueteaming we hope that companies and individuals will strive to include this in their systems and create safer LLMs.
