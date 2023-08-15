# eco4cast examples 

Here you can find several examples of using eco4cast package. 

There two main options how to use eco4cast: 
- Use it with Google Cloud 
- Use it on your local machine

## Usage example with Google Cloud
You can find notebook on using eco4cast with Google Cloud in eco4cast_demo folder or using a link [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIRI-Institute/eco4cast/blob/main/examples/eco4cast_demo/quick_start_guide.ipynb)

This notebook contains a step-by-step guide on creating a virtual machine, signing in on electricitymaps.com and necessary Linux shell commands.

Also there are two folders MNIST_google_cloud and ImageNet_google_cloud contating code to train models for these datasets. You can start them similarily to Jupyter notebook.

### Example files details
Here we will describe what is going on in examples files `master_machine_main.py` and `vm_main.py` and how you can change them for your needs.

#### master_machine_main.py
Basically this file consists of usage of just one class - Controller class. This class's main functions ois to start training on Google Cloud VM. It uses ssh to connect to it (so you nhave to pass different ssh parameters). It generates training intervals using CO2Predictor (neural net to get 24 h forecast of CO2 in 13 regions) and IntervalGenerator to deal with the forecast. This class also uses Google Cloud API to move VM between zones to  get minimal value of CO2 emission at the time. 

#### vm_main.py
This file consists of all the training process logic. Firstly it initializes pytorch model with pytorch datasets. After that you can specify some callbacks you want to use during process (callbacks are realized using Lighting Fabric). Argument parser is needed to get information about training periods in current zone (this is a List of Tuples of form [(start_time, end_time, interval_emission)]). 

The main part of this file is IntervalTrainer class. This class uses custom pytorch logic so it can stop and resume training process on different VMs without losing any information. It even saves current batch info. So if your model has a large epoch time you can start it in one Google Cloud zone and continue it in another. 

Of course you can modify `vm_main.py` as you want. Probably you will use your own dataset, so you have to load it to VM once and import it in `vm_main.py`.


## Usage example on your local machine
You can find notebook on using eco4cast on your local machine in eco4cast_local_demo folder or using a link [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIRI-Institute/eco4cast/blob/main/examples/eco4cast_local_demo/local_quick_start_guide.ipynb). 

This notebook containg step-by-step guide on how to train Pytorch model during time with minimal CO2 emission with your own hardware. 

### Available electricitymaps zones to work locally: 
- "BR-CS" (Central Brazil)
- "CA-ON" (Canada Ontario) 
- "CH" (Switzerland) 
- "DE" (Germany) 
- "PL" (Poland) 
- "BE" (Belgium) 
- "IT-NO" (North Italy) 
- "CA-QC" (Canada Quebec) 
- "ES" (Spain) 
- "GB" (Great Britain) 
- "FI" (Finland) 
- "FR" (France) 
- "NL" (Netherlands)
