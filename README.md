# bipedal-methods
Supplementary code for Chapter 5 of my PhD thesis at Keele University, Neuroevolution of Bipedal Locomotion: algorithmic, balance penalty and morphological improvements for improved robustness and performance.

The "deep-neuroevolution" directory contains a modified version of the code featured in Uber AI's deep-neuroevolution repo: https://github.com/uber-research/deep-neuroevolution used in Chapter 5. Instructions on how to run the code and visualise results snapshots produced on Linux are featured on the readme there. The additional files in the "others" directory are taken from the openai gym installation required by the deep-neruoevolution repo, and should be substituted in gym/envs/mujoco/humanoid.py and  gym/envs/mujoco/assets/humanoid.xml respectively. All sections edited by me are labelled with comments beginning BJ. Principal modifications include:

1-Control cost multiplier

This is a multiplier applied to the reward function of the humanoid environment. Activated through policies.py at a given timestep, when the Flipper boolean in humanoid.py is applied, a 0.25 multiplier is applied to the control cost term.

2-Radial Fall Constraint

When the circlecheck boolean in humanoid.py is activated, the radial fall constraint (failure if com is outside the circle drawn by the feet with a certain radius) will be enabled with the multiplier m, set in policies.py.

3-Action noise

A comment in the viz.py file shows where the ac.noise_std variable can be modified to view gaits under action noise.
 
Finally a selection of evolved gaits are included in the "gaits" directory, representing gaits produced by the default, the rp75, the s500 and the combo permutations. Another four are included with the same gaits featuring an action noise parameter of 0.4. 

CHANGELIST
==========
deep-neuroevolution/es_distributed/ es.py 
  -added a limiter to keep to 600 iterations
  -set snapshots to be added to folder
deep-neuroevolution/es_distributed/ policies.py
  -included settings parameters to adjust the two main modifications
deep-neuroevolution/scripts/ viz.py
  -signposted action noise adjustment settings 
others/ humanoid.py
	-the bulk of the code for the control cost multiplier and radial fall constraint is located here
others/ humanoid.xml
	-the floor texture and shadows are adjusted for clarity of visualisatoin

(Any issues contact me at benjack795@gmail.com)
