# SmartScheduler

+ [About SmartScheduler :clipboard:](#1)
+ [Installation :wrench:](#2)
+ [Usage example (Tutorial on training with MNIST) :computer:](#3)
+ [How to use package without Google Cloud](#4)
+ [Citing](#5)
+ [Feedback :envelope:](#6) 



## About SmartScheduler :clipboard: <a name="1"></a> 
This package is designed to reduce CO2 emissions while training neural network models. The main idea of the package is to run the learning process at certain time intervals on certain Google Cloud servers with minimal emissions. A neural network (TCN) trained on the historical data of 13 zones is used to predict emissions for 24 hours ahead.

Currently supported Google Cloud zones: 'southamerica-east1-b', 'northamerica-northeast2-b', 'europe-west6-b', 'europe-west3-b', 'europe-central2-b', 'europe-west1-b', 'europe-west8-a', 'northamerica-northeast1-b', 'europe-southwest1-c', 'europe-west2-b', 'europe-north1-b', 'europe-west9-b',  'europe-west4-b' .

## Installation <a name="2"></a> 
Package can be installed using Pypi:
```
pip install smartscheduler
```

## Usage example. Tutorial on training with MNIST <a name="3"></a>
### What you will need
- Google Cloud account
- ElectricityMaps account with free trial
- Master machine to control VMs 

### Step 1. Setting up master machine.
Make a project directory and create venv (or conda env). Do necessary installations:
```
python3 -m venv venv
source venv/bin/activate
pip install smartscheduler
```

### Step 2. Google cloud setup
To setup you will need to go on [Google Cloud Console](console.cloud.google.com) and create project. Choose your project and click on "Activate Cloud Shell" (top-right corner of the window). Do following steps:

```
gcloud auth application-default login
```

Download your `application_default_credentials.json` and place it in project folder on Master machine.

Also you need to setup your SSH key for project. You can do it [here](https://console.cloud.google.com/compute/metadata/sshKeys).

### Step 3. Electricity Maps setup
Go to the [electricitymaps website](api-portal.electricitymaps.com) and create an account. Apply for free trial period and copy your API key (primary) into electricitymaps_api.py.


### Step 4. Creating VM
Create VM in "Compute Engine" section on Google Cloud. Select configutation and OS (for this tutorial we used E2-medium VM with 25 GB disk and Ubuntu Minimal 22.10). Set your SSH key (In this example VM user is named "scheduler") in Security settings of VM to be able to connect to it. We created it in "northamerica-northeast1-b" zone.


You will probably need to install extra dependencies on VM:
```
sudo apt update
sudo apt install python3.10-venv 
```

### Step 5. Create venv on VM
Connect to VM SSH and do folowing steps.
```
python3 -m venv venv
source venv/bin/activate
pip install smartscheduler
pip install torchvision  # Installation needed for MNIST
```

### Step 6. Create folder for python scripts
```
mkdir scheduler_task
```

### Step 7. Edit vm_main.py for your purposes.
Download file vm_main.py from github (can be found in `examples` folder).
This is the main file which includes all the training process logic. Here you can choose what callbacks will be used, what kind of model, dataset and all the parameters. 


### Step 8. Copy vm_main.py to VM
```
scp vm_main.py scheduler@your_ip:scheduler_task/
```


### Step 9. Run task on your Master machine
Download example `master_machine_main.py` from examples folder on our github. Edit some VM info in file (current ip adress, zone, your project name, instance name).
And after that you are ready to start the training!
```
python master_machine_main.py
```



### Example files details
Here we will describe what is going on in examples files `master_machine_main.py` and `vm_main.py` and how you can change them for your needs.

#### master_machine_main.py
Basically this file consists of usage of just one class - Controller class. This class's main functions ois to start training on Google Cloud VM. It uses ssh to connect to it (so you nhave to pass different ssh parameters). It generates training intervals using CO2Predictor (neural net to get 24 h forecast of CO2 in 13 regions) and IntervalGenerator to deal with the forecast. This class also uses Google Cloud API to move VM between zones to  get minimal value of CO2 emission at the time. 


#### vm_main.py
This file consists of all the training process logic. Firstly it initializes pytorch model with pytorch datasets. After that you can specify some callbacks you want to use during process (callbacks are realized using Lighting Fabric, thats why you can't just take Lightning-pytorch callbacks). Argument parser is needed to get information about training periods in current zone (this is a List of Tuples of form [(start_time, end_time)]). 

The main part of this file is IntervalTrainer class. This class uses custom pytorch logic so it can stop and resume training process on different VMs without losing any information. It even saves current batch info. So if your model has a large epoch time you can start it in one Google Cloud zone and continue it in another. 

Of course you can modify `vm_main.py` as you want. Probably you will use your own dataset, so you have to load it to VM once and import it in `vm_main.py`.


## How to use package without Google Cloud. <a name="4"></a>
If you want to use our scheduler without Google Cloud VMs and you are in one of the available zones you can use `local_main.py` example and specify your electricitymaps zone in code. Scheduler will start training only during time with minimal CO2 emission.

Available electricitymaps zones: "BR-CS", "CA-ON", "CH", "DE", "PL", "BE", "IT-NO", "CA-QC", "ES", "GB", "FI", "FR", "NL"

## Citing <a name="5"></a>
Paper info


## Feedback <a name="6"></a>
email?