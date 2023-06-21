from google_cloud_vm_moving import google_cloud_move_vm
import datetime
import paramiko
from time import sleep

# Make sure venv is installed and located in $HOME$/scheduler/venv on VM
# Make sure ssh is anabled on VM and ssh key of this machine is in .ssh/authorized_keys of VM.  Username is scheduler
# Make sure your vm_main.py is located in $HOME$/scheduler/scheduler_task/vm_main.py on VM with


def setup_ssh_execution(
    ip, ssh_port, username="scheduler", command=None, command_arguments=""
):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, port=ssh_port, username=username)

    command = (
        "/home/scheduler/venv/bin/python /home/scheduler/scheduler_task/vm_main.py"
    )
    command += " " + command_arguments

    transport = ssh_client.get_transport()
    bufsize = -1
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)
    stdin = channel.makefile_stdin("wb", bufsize)
    stdout = channel.makefile("r", bufsize)
    stderr = channel.makefile_stderr("r", bufsize)

    return channel, stdout


# Should be used from some IntervalPredictor
training_intervals = [
    (
        "europe-southwest1-a",
        [
            (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=0),
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=30),
            ),  # interval start_time, end_time
            (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=50),
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=80),
            ),
        ],
    ),
    (
        "us-west1-b",
        [
            (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(minutes=7, seconds=0),
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(minutes=10, seconds=0),
            )
        ],
    ),
]

# current_zone = "us-west1-b"
current_zone = "europe-southwest1-a"
current_instance_name = "vm"
project_id = "test-smart-scheduler"
current_ip = "34.175.199.46"
ssh_port = 22
load_states = False

for zone, time_intervals in training_intervals:
    if zone != current_zone:
        # new_instance = google_cloud_move_vm(
        #     current_zone=current_zone, project_id=project_id, new_zone=zone
        # )
        print("Moving")
        load_states = True
        # current_ip = new_instance.network_interfaces[0].access_configs[0].nat_i_p

    # Scheduling job 10 seconds before first start_time
    while datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=3
    ) < time_intervals[0][0] - datetime.timedelta(seconds=30):
        sleep(1)

    argument_intervals = [
        (s.strftime("%Y%m%d%H%M%S"), e.strftime("%Y%m%d%H%M%S"))
        for s, e in time_intervals
    ]
    argument_intervals = [item for t in argument_intervals for item in t]
    argument_intervals = ",".join(argument_intervals)

    command_arguments = f"--times {argument_intervals}"
    if load_states:
        command_arguments += " --load_states"
    print("SSH Starting")
    channel, stdout = setup_ssh_execution(
        current_ip, ssh_port, command_arguments=command_arguments
    )

    try:
        for line in iter(stdout.readline, ""):
            print(line, end="")
    except KeyboardInterrupt:
        channel.send("\x03")  # Ctrl-C
        for line in iter(stdout.readline, ""):
            print(line, end="")
        print("Interrupted")
        break


print("Successfully finished all intervals")
