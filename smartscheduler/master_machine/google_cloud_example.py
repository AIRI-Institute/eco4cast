from smartscheduler.master_machine.google_cloud_vm_moving import google_cloud_move_vm
import datetime
import paramiko
import time
from smartscheduler.master_machine.utils import codes_to_gcloud_zones
from smartscheduler.master_machine.compute.client_library.snippets.instances.stop import (
    stop_instance,
)
from time import sleep
from smartscheduler.master_machine.interval_predictor import CO2Predictor, IntervalGenerator
from smartscheduler.master_machine.utils import code_names


# Make sure ssh is anabled on VM and ssh key of this machine is in .ssh/authorized_keys of VM.  Username is scheduler

# Make sure venv is created on VM
# Necessary: pip install apscheduler torch lightning eco2ai
# For this example also:  pip install torchvision


def setup_ssh_execution(
    ip,
    ssh_port,
    python_path,
    vm_main_path,
    username="scheduler",
    command=None,
    command_arguments="",
):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, port=ssh_port, username=username)

    vm_main_folder = "/".join(vm_main_path.split("/")[:-1])
    vm_main_file = vm_main_path.split("/")[-1]

    command = f"cd {vm_main_folder}; {python_path} {vm_main_file}"
    command += " " + command_arguments

    transport = ssh_client.get_transport()
    bufsize = -1
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)
    # stdin = channel.makefile_stdin("wb", bufsize)
    stdout = channel.makefile("r", bufsize)
    # stderr = channel.makefile_stderr("r", bufsize)

    return channel, stdout



username = "scheduler"
python_path = f"/home/{username}/venv/bin/python"
vm_main_path = f"/home/{username}/scheduler_task/vm_main.py"
current_ip = "34.152.12.159"
ssh_port = 22


current_zone = "northamerica-northeast1-b"
current_instance_name = "instance-1"
project_id = "test-smart-scheduler"
load_states = False
intervals_prediction_period = 3600  # seconds


max_emission_value = 130
min_interval_size = 1
co2_delta_to_move = 0
last_prediction_time = datetime.datetime.now()
co2_predictor = CO2Predictor()
interval_generator = IntervalGenerator(code_names)
co2_forecast = co2_predictor.predict_co2()
predicted_intervals = interval_generator.generate_intervals(
    forecasts=co2_forecast,
    max_emission_value=max_emission_value,
    co2_delta_to_move=co2_delta_to_move,
    min_interval_size=min_interval_size,
    # exclude_zones=['FR'],
    # include_zones=['BR-CS']
)

# predicted_intervals = [
#     (
#         "CA-QC",
#         [
#             (
#                 datetime.datetime(2023, 7, 11, 10, 0, tzinfo=datetime.timezone.utc),
#                 datetime.datetime(2023, 7, 11, 10, 20, tzinfo=datetime.timezone.utc),
#             )
#         ],
#     ),
#     (
#         "FR",
#         [
#             (
#                 datetime.datetime(2023, 7, 11, 10, 30, tzinfo=datetime.timezone.utc),
#                 datetime.datetime(2023, 7, 12, 8, 0, tzinfo=datetime.timezone.utc),
#             )
#         ],
#     )
# ]


interval_idx = 0
shutting = False
while interval_idx < len(predicted_intervals) and not shutting:
    code, time_intervals = predicted_intervals[interval_idx]
    zone = codes_to_gcloud_zones[code]
    if zone != current_zone:
        print("Moving VM")
        start_moving_time = time.time()
        new_instance = google_cloud_move_vm(
            current_zone=current_zone,
            current_instance_name=current_instance_name,
            project_id=project_id,
            new_zone=zone,
        )
        print(f"Moving completed in {int(time.time()-start_moving_time)} seconds")
        current_ip = new_instance.network_interfaces[0].access_configs[0].nat_i_p
        current_zone = zone
        sleep(10)

    # Scheduling job 30 seconds before first start_time
    while datetime.datetime.now(datetime.timezone.utc) < time_intervals[0][
        0
    ] - datetime.timedelta(seconds=30):
        time.sleep(1)

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
        current_ip,
        ssh_port,
        python_path,
        vm_main_path,
        username=username,
        command_arguments=command_arguments,
    )

    try:
        for line in iter(stdout.readline, ""):
            print(line, end="")
            if "Traceback" in line:
                pass
            if "End of training." in line:
                print("Stopping master machine")
                shutting = True
                break

    except KeyboardInterrupt:
        channel.send("\x03")  # Ctrl-C
        for line in iter(stdout.readline, ""):
            print(line, end="")
        print("Interrupted")
        break
    interval_idx += 1
    if datetime.datetime.now() > last_prediction_time + datetime.timedelta(
        seconds=intervals_prediction_period
    ):
        last_prediction_time = datetime.datetime.now()
        co2_forecast = co2_predictor.predict_co2()
        predicted_intervals = interval_generator.generate_intervals(
            forecasts=co2_forecast,
            max_emission_value=max_emission_value,
            co2_delta_to_move=co2_delta_to_move,
            min_interval_size=min_interval_size,
            # exclude_zones=['FR'],
            # include_zones=['BR-CS']
        )
        interval_idx = 0

    load_states = True


print("Successfully finished all intervals. Stopping Instance")
stop_instance(
    project_id=project_id, zone=current_zone, instance_name=current_instance_name
)
