# SmartScheduler

+ [About SmartScheduler :clipboard:](#1)
+ [Installation :wrench:](#2)
+ [Usage example (Tutorial on training with MNIST) :computer:](#3)
+ [Citing](#4)
+ [Feedback :envelope:](#5) 



## About SmartScheduler :clipboard: <a name="1"></a> 
This package is designed to reduce CO2 emissions while training neural network models. The main idea of the package is to run the learning process at certain time intervals on certain Google Cloud servers with minimal emissions. A neural network (TCN) trained on the historical data of 13 zones is used to predict emissions for 24 hours ahead.

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
pip instal smartscheduler
pip install torchvision  # Installation needed for MNIST
```

### Step 6. Create folder for python scripts
```
mkdir scheduler_task
```

### Step 7. Edit vm_main.py for your purposes.
Download file vm_main.py from github (can be found in `examples` folder).
This is the main file which includes all the training process logic. Here you can choose what callbacks will be used, what kind of model, dataset and all the parameters. 


### Step 7. Copy vm_main.py to VM
```
scp vm_main.py scheduler@your_ip:scheduler_task/
```


### Step 8. Run task on your Master machine
Download example `master_machine_main.py` from examples folder on our github. Edit some VM info in file (current ip adress, zone, your project name, instance name).
And after that you are ready to start the training!
```
python master_machine_main.py
```


### Some extra info
Due to paramiko package restrictions tqdm progress bar can't be shown during process. It only show when the epoch (training or validation) is finished.


## Citing <a name="4"></a>
Paper info


## Feedback <a name="5"></a>
email?