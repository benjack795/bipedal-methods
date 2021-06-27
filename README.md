# bipedal-methods
Supplementary code for Chapter 5 of Neuroevolution of Bipedal Locomotion: algorithmic, balance penalty and morphological improvements for improved robustness and performance

The folder "deep-neuroevolution" contains a modified version of the code featured in Chapter 5 of Neuroevolution of Bipedal Locomotion: algorithmic, balance penalty and morphological improvements for improved robustness and performance

Also included are modified files from the OpenAI gym github used alongside ("other" folder). These include xml files (add to gym/envs/mujoco/assets) for the default and rough ground environments, as well as a modified humanoid.py (add to gym/envs/mujoco) for the changes to the humanoid environment to enable the methods employed. By replacing appropriately in your deep-neuroevolution installation and your openai-gym installation you should be able to reproduce this work. (If not contact me, benjack795@keele.ac.uk)

Finally a selection of evolved gaits are included ("gaits" folder) representing gaits produced by the default, the rp75, the s500 and the combo permutations. Another four are included with the same gaits featuring an action noise parameter of 0.4. The final gait features the s500 runs, which performed best on rough ground, walking on unstable terrain at 3% of the height of the walker. 

MODIFICATIONS

1-Delayed multiplier

This is a multiplier applied to the reward function of the humanoid environment. Activated through policies.py, when the Flipper boolean in humanoid.py is applied, the reward function is changed.

2-Radial Fall Constraint

When the circlecheck boolean in humanoid.py is activated, the radial fall constraint (failure if com is outside the circle drawn by the feet with a certain radius) will be enabled with the multiplier m, set in policies.py.

3-Action noise testing

A comment in the viz.py file shows where the ac.noise_std variable can be modified to view gaits under action noise.

4-Environment noise testing

The humanoid_hfield.xml features a hfield that can be edited (alongsides the walkers starting height) to view gaits against environmental noise.


