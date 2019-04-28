# Simple Reinforcment Learning Implementations
Implementations of popular reinfocment learning algorithms integrated with OpenAI Gym. Meant to be easy to understand.

## Prerequisites
```
gym==0.12.1
tensorflow==1.13.1
numpy==1.16.2
scipy==1.2.1
```

## Getting Started
To get started first clone the repo localy. Currently the algorithms implemented are A2C and PPO. To run one of there simply go into its respective folder and run the main.py file.

```
python main.py
```
The agent should begin to train automatically and a non training agent will render the game with an up to date network. 

## Configurations
Flags are not yet implemented, to change the enironment or number of parallel agents simply change the corresponding variables in the main.py file.

## Authors
* **Pablo Beltran** - [Sturdyplum](https://github.com/Sturdyplum)
* **Alex Coleman** - [alexrcoleman](https://github.com/alexrcoleman)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
