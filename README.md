# MADDPG PYTORCH Implementation
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm is implemented in pytorch via Unity ML-Agents Toolkit. Link to the paper https://arxiv.org/pdf/1706.02275.pdf

## Project Description
This is the sample Reinforcement Learning project which uses MADDPG(Multi-agent deep deterministic policy gradient) algorithm to learn competitive behaviour in the given unity environment.The agents were trained based on curriculum learning approach along with self-play to improve its performance.
The main aim of the both teams is to return the opponent flag to its base zone while throwing ball to the opponent to maximize its reward objectives.For more info visit [documentation](https://github.com/love481/DodgeBall_MA_RL/blob/1b5be765bf176dfee7ba35f6a55d8bd9ee6343bc/final_report.pdf).

https://user-images.githubusercontent.com/54012619/168807315-bf5ccde2-d5ca-443a-b3d6-317b4c645dcb.mp4

## Requirements:
* Python >= 3.8.10
* torch >=1.8.1+cpu
* mlagents >= 0.28.0
* mlagents_envs >= 0.28.0
* numpy
* matplotlib
* math

## Installation
Run this command on command prompt to clone the repository


`git clone https://github.com/love481/DodgeBall_MA_RL.git`

## Running code
To train or evaluate the models run on command line

`python main.py`

## Code Structure
* `\scripts\main.py` --> To start training or testing of models.Use evaluate == false to start training else evaluating.
* `\scripts\common\...` --> Contains code for replay buffer, argumenents and noises
* `\scripts\agent.py` --> Actions to each agent
* `\scripts\runner.py` --> Integrating all modules together to run the program.
* `\scripts\maddpg\actor_critic.py` --> Pytorch implementation of actor and critic part
* `\scripts\maddpg\agents.py` --> Interacts with unity environment via ML-agent toolkit
* `\scripts\maddpg\maddpg.py` --> Implementation modules of maddpg algorithm
* `\scripts\scratch\..` --> All trained model learning from scratch
* `\scripts\curriculum\..` --> All trained model using curriculum learning
* `\scripts\train_test_data\..` --> All ploted graph


## Results
### Training
![image](https://user-images.githubusercontent.com/54012619/168816662-f9380d53-548f-42aa-a61d-6ac65e92667b.png)

### Testing
![image](https://user-images.githubusercontent.com/54012619/168816416-d5573749-34bc-471f-95cc-cf1b519cac10.png)

## Contact Information
Please feel free to contact me if any help needed

Email: *075bei016.love@pcampus.edu.np*

