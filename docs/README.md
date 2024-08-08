# Koopman-RL Project Document
Last updated 8 August 2024

This project centers around learning Koopman-inspired control policies (mainly for robotic applications) using RL techniques. We investigate the feasibility of Koopman-structured policies for robotic settings.


## Research Questions
1) Similar to the idea of motion and pose primitives, can we extract a set of manipulation primitives, and then use them to perform skill transfer between tasks? That is, instead of learning a manipulation problem by its control inputs, can we abstract out a basis of motion primitives from training tasks and compose primitives together to perform new tasks with high fidelity?

    - This is the latestage of this project. We are not here yet.

2) Can we learn the koopman matrix through RL? 

    - Yes. Experiments done with this repo show that RL algorithms are sufficient to train Koopman-operator based policies. Policies for various robotic control problems have been successfully trained with Augmented Random Search (ARS) and Proximal Policy Optimization (PPO) algorithms.

3) Is it beneficial in any way to represent a policy in a koopman-inspired format?

    - Maybe. Right now, we are investigating if we can learn with koopman policies in a way that is (1) more expressive than linear policies (which it should be), and (2) if this compressed representation relative to other policies (like full MLP policies) would train faster. We suspect that the answer to (2) is yes, but with a heavy dependency on the lifting function used. 