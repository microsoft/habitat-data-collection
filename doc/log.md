# Logs


#### 11/3/2022 Conda env for habitat

When you conda install `habitat-sim` and export the environment to `env.yml`, `habitat-sim` will appear under the section `pip`. While `habitat-sim` does not have distribution in `pip`, which causes you error when you try to create environment from `env.yml`. The solution is to 
manually:
* delete `habitat-sim` under the `pip` section
* add `habitat-sim` with approriate version.

https://anaconda.org/aihabitat/habitat-sim/files lists all available packages, choose the one that is consistent with configuration at installtion.


#### 11/3/2022 habitat-lab top_down_map bug

Fork the habitat-lab and repo and fix the bug on forked repo. Fork the habitata repo and install it inside docker


#### 11/25 semantic sensor

semantic ensor can only be used along with rgb (not alone).

semantic sensor done. ~1/8 scenes come with semantic annotation. May need to generate task config files by ourselves in the future to use semantic annotation.

scene config file path in object nav is need to be changed

#### 11/28

noise mode: Gaussion noise for rgb and RedwoodDepthNoiseModel for depth image 


#### 1/19/2023

profile code 

#### 1/24/2023

removing raidius setup in agent and check shortest path.points (not None or empty) to avoid 1-step episodes.


Saving data locally halves the time compared to saving to blobfuse.

habitat-lab using gpu without specifying gpu in config. 4 workers (processes) using 4Gb GPU memory. Around 1Gb GPU memory per process


#### 1/25/2023

habitat hm3d semantic segmentation is instance level (not class level)

The default radius = 0.1
