# habitat-simulation

Run agents in habitat simulation enviroment (habitat-lab), and collect data for training.

## Run Locally 
First install conda environment in `docker/`:
* `env.yaml` contains habitat and pytorch lightning
* `env_habitat.yaml` contains only habitat (more friendly maybe)

Then install habitat-lab locally:
```
git clone https://github.com/shuhangchen/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
```

Install utilities packages [here](https://github.com/AutonomousSystemsResearch/utils_package/tree/shc).
## run on aml
The current `jobs/amlt.yaml` is already set up for aml. Note that the docker image at `docker.io:shuhangchen/habitat:latest` contains both habitat (sim and lab) and pytorch lightning, and it does not depend on the conda environments in `docker`. 

