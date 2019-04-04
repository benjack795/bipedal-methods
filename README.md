# bipedal-methods
supplementary code for "Deep Neuroevolution of Humanoids that Walk Further and Faster with Robust Gaits"

The folder "deep-neuroevolution" contains a modified version of the code featured on the UberAI github for deep-neuroevolved humanoid walking.

Also included are modified files from the OpenAI gym github ("other" folder). These include xml files (located in gym/envs/mujoco/assets) for the default and rough ground environments, as well as a modified humanoid.py (located in gym/envs/mujoco) for the changes to the humanoid environment to enable the methods employed.

Finally a selection of evolved gaits is included representing gaits produced by the default, the rp75, the s500 and the combo permutations. Another four are included with the same gaits featuring an action noise parameter of 0.4. The final gait features the s500 runs, which performed best on rough ground, walking on unstable terrain up to 3% of the height of the walker. 

MODIFICATIONS

1-Delayed multiplier

2-Radial Fall Constraint

3-Action noise testing

4-Environment noise testing
