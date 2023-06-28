# Tutorial on starting MNIST interval training
## What you will need
- Google Cloud account
- Master machine to control VMs 

## Step 1. Google cloud setup
To setup you will need to go on [Google Cloud Console](console.cloud.google.com) and create project. Choose your project and click on "Activate CLoud Shell" (top-right corner of the window). Do following steps:

```
gcloud auth application-default login
```

Download your `application_default_credentials.json` and place it in project folder on Master machine.

## Step 2. Creating VM
Create VM in "Compute Engine" section on Google Cloud. Select configutation and OS (for this tutorial we used E2-medium VM with 25 GB disk and Ubuntu Minimal 22.10). Set your SSH key (In this example VM user is named "scheduler") in Security settings of VM to be able to connect to it. We created it in "europe-southwest1-a" zone.


You will probably need to install extra dependencies on VM:
```
sudo apt update
sudo apt install python3.10-venv
```

## Step 3. Create venv on VM
Connect to VM SSH manually and do  folowing steps.
```
python3 -m venv venv
source venv/bin/activate
pip install apscheduler torch lightning eco2ai  # pip instal smartscheduler[vm] in future
pip install torchvision  # Installation needed for MNIST
```

## Step 4. Create folder and place 
```
mkdir scheduler_task
cd  scheduler_task
```


## Step 5. Copy files to VM
```
scp smart_scheduler_concept.py  scheduler@your_ip:scheduler_task/
scp vm_main.py  scheduler@your_ip:scheduler_task/
scp utils.py  scheduler@your_ip:scheduler_task/
```

## Step 6. Run task on your Master machine
Edit some VM info in file (current ip adress, zone, your project name, instance name).
Make sure there is `google_cloud_moving.py` and `compute` folder (can be downloaded [here](https://github.com/GoogleCloudPlatform/python-docs-samples)) are in your project directory on master machine. Then you are ready to run!

```
python google_cloud_example.py
```