# bipedal-methods
supplementary code for "Deep Neuroevolution of Humanoids that Walk Further and Faster with Robust Gaits"

The folder "deep-neuroevolution" contains a modified version of the code featured on the UberAI github for deep-neuroevolved humanoid walking. Also included are modified files from the OpenAI gym github used alongside ("other" folder). These include xml files (located in gym/envs/mujoco/assets) for the default and rough ground environments, as well as a modified humanoid.py (located in gym/envs/mujoco) for the changes to the humanoid environment to enable the methods employed. By replacing appropriately in your deep-neuroevolution installation and your openai-gym installation you should be able to reproduce this work. (If not contact me.)

Finally a selection of evolved gaits is included representing gaits produced by the default, the rp75, the s500 and the combo permutations. Another four are included with the same gaits featuring an action noise parameter of 0.4. The final gait features the s500 runs, which performed best on rough ground, walking on unstable terrain at 3% of the height of the walker. 

MODIFICATIONS

1-Delayed multiplier

This is a multiplier applied to the reward function of the humanoid environment. Activated through policies.py, when the Flipper boolean in humanoid.py is applied, the reward function is changed.

2-Radial Fall Constraint

When the circlecheck boolean in humanoid.py is activated, the radial fall constraint (failure if com is outside the circle drawn by the feet with a certain radius) will be enabled with the multiplier m, set in policies.py.

3-Action noise testing

A comment in the viz.py file shows where the ac.noise_std variable can be modified to view gaits under action noise.

4-Environment noise testing

The humanoid_hfield.xml features a hfield that can be edited (alongsides the walkers starting height) to view gaits against environmental noise.


