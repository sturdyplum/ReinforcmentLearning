# Simple Reinforcment Learning Implementations
Implementations of popular reinfocment learning algorithms integrated with OpenAI Gym. Meant to be easy to understand.

## Setup
Install the required libraries. First do:
```
pip install -r requirements.txt
```

Then you will need to install another library to run atari-py. A forked version that includes Windows support is used. To install it, run the command:
```
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
```

## Running
To get started first clone the repo locally. Currently the algorithms implemented are A2C and PPO. To run one of there simply go into its respective folder and run the main.py file.

```
python main.py
```
The agent should begin to train automatically and a non training agent will render the game with an up to date network. 

## Configurations
Flags are not yet implemented, to change the environment or number of parallel agents simply change the corresponding variables in the main.py file.

## Authors
* **Pablo Beltran** - [Sturdyplum](https://github.com/Sturdyplum)
* **Alex Coleman** - [alexrcoleman](https://github.com/alexrcoleman)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
