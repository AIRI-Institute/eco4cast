# SmartScheduler Liblrary
## Some description









# Tutorial on starting MNIST interval training
## What you will need
- Google Cloud account
- ElectricityMaps account with free trial
- Master machine to control VMs 

## Step 1. Setting up master machine.
Make a project directory and create venv (or conda env). Do necessary installations:
```
python3 -m venv venv
source venv/bin/activate
pip install smartscheduler[mastermachine]
```

## Step 2. Google cloud setup
To setup you will need to go on [Google Cloud Console](console.cloud.google.com) and create project. Choose your project and click on "Activate Cloud Shell" (top-right corner of the window). Do following steps:

```
gcloud auth application-default login
```

Download your `application_default_credentials.json` and place it in project folder on Master machine.

Also you need to setup your SSH key for project. You can do it [here](https://console.cloud.google.com/compute/metadata/sshKeys).

# Step 3. Electricity Maps setup
Go to the [electricitymaps website](api-portal.electricitymaps.com) and create an account. Apply for free trial period and copy your API key (primary) into electricitymaps_api.py.


## Step 4. Creating VM
Create VM in "Compute Engine" section on Google Cloud. Select configutation and OS (for this tutorial we used E2-medium VM with 25 GB disk and Ubuntu Minimal 22.10). Set your SSH key (In this example VM user is named "scheduler") in Security settings of VM to be able to connect to it. We created it in "northamerica-northeast1-b" zone.


You will probably need to install extra dependencies on VM:
```
sudo apt update
sudo apt install python3.10-venv 
```

## Step 5. Create venv on VM
Connect to VM SSH and do folowing steps.
```
python3 -m venv venv
source venv/bin/activate
pip instal smartscheduler[virtualmachine] 
pip install torchvision  # Installation needed for MNIST
```

## Step 6. Create folder for python scripts
```
mkdir scheduler_task
```

## Step 7. Edit vm_main.py for your purposes.
Download file vm_main.py from github.
This is the main file which includes all the training process logic. Here you can choose what callbacks will be used, what kind of model, dataset and all the parameters. 


## Step 7. Copy vm_main.py to VM
```
scp vm_main.py scheduler@your_ip:scheduler_task/
```


## Step 8. Run task on your Master machine
Edit some VM info in google_cloud_example.py file (current ip adress, zone, your project name, instance name).
And after that you are ready to start the training!
```
python google_cloud_example.py
```


## Some extra info
Due to paramiko package restrictions tqdm progress bar can't be shown during process. It only show when the epoch (training or validation) is finished.