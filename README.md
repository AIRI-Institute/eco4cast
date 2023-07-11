# Tutorial on starting MNIST interval training
## What you will need
- Google Cloud account
- ElectricityMaps account with free trial
- Master machine to control VMs 

## Step 1. Google cloud setup
To setup you will need to go on [Google Cloud Console](console.cloud.google.com) and create project. Choose your project and click on "Activate Cloud Shell" (top-right corner of the window). Do following steps:

```
gcloud auth application-default login
```

Download your `application_default_credentials.json` and place it in project folder on Master machine.

Also you need to setup your SSH key for project. You can do it [here](https://console.cloud.google.com/compute/metadata/sshKeys).

# Step 2. Electricity Maps setup
Go to the [electricitymaps website](api-portal.electricitymaps.com) and create an account. Apply for free trial period and copy your API key (primary) into electricitymaps_api.py.


## Step 3. Creating VM
Create VM in "Compute Engine" section on Google Cloud. Select configutation and OS (for this tutorial we used E2-medium VM with 25 GB disk and Ubuntu Minimal 22.10). Set your SSH key (In this example VM user is named "scheduler") in Security settings of VM to be able to connect to it. We created it in "northamerica-northeast1-b" zone.


You will probably need to install extra dependencies on VM:
```
sudo apt update
sudo apt install python3.10-venv 
```

## Step 4. Create venv on VM
Connect to VM SSH and do folowing steps.
```
python3 -m venv venv
source venv/bin/activate
pip install apscheduler torch lightning eco2ai  # pip instal smartscheduler[vm] in future
pip install torchvision  # Installation needed for MNIST
```

## Step 5. Create folder for python scripts
```
mkdir scheduler_task
```


## Step 6. Copy files to VM
```
scp smart_scheduler_concept.py vm_main.py utils.py scheduler@your_ip:scheduler_task/
```


## Step 7. Run task on your Master machine
Edit some VM info in file (current ip adress, zone, your project name, instance name).
Make sure there is `google_cloud_moving.py` and `compute` folder (can be downloaded [here](https://github.com/GoogleCloudPlatform/python-docs-samples)) are in your project directory on master machine. Then you are ready to run!

```
python google_cloud_example.py
```