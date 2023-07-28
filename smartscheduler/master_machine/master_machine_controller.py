import paramiko
import os
import datetime
from smartscheduler.master_machine.interval_predictor import (
    CO2Predictor,
    IntervalGenerator,
)
from time import sleep
from smartscheduler.master_machine.compute.client_library.snippets.instances.stop import (
    stop_instance,
)
from smartscheduler.master_machine.google_cloud_vm_moving import google_cloud_move_vm
from smartscheduler.master_machine.utils import codes_to_gcloud_zones
import time


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
    ssh_client.connect(hostname=ip, port=ssh_port, username=username, timeout=10)

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


class Controller:
    def __init__(
        self,
        credentials_path,
        ssh_username,
        ssh_python_path,
        ssh_vm_main_path,
        current_vm_ip,
        ssh_port,
        current_zone,
        current_instance_name,
        project_id,
        intervals_prediction_period,
        co2_predictor: CO2Predictor,
        interval_generator: IntervalGenerator,
    ):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

        self.load_states = False
        self.last_prediction_time = datetime.datetime.now()
        self.co2_predictor = co2_predictor
        self.co2_forecast = self.co2_predictor.predict_co2()
        self.interval_generator = interval_generator

        (
            self.predicted_intervals,
            self.zone_indices,
        ) = self.interval_generator.generate_intervals(
            forecasts=self.co2_forecast,
        )

        self.ssh_username = ssh_username
        self.ssh_python_path = ssh_python_path
        self.ssh_vm_main_path = ssh_vm_main_path
        self.current_vm_ip = current_vm_ip
        self.ssh_port = ssh_port
        self.current_zone = current_zone
        self.current_instance_name = current_instance_name
        self.project_id = project_id
        self.intervals_prediction_period = intervals_prediction_period

        # FOR TESTING
        # self.predicted_intervals = [
        #     # (
        #     #     "FR",
        #     #     [
        #     #         (
        #     #             datetime.datetime(2023, 7, 13, 5, 0, tzinfo=datetime.timezone.utc),
        #     #             datetime.datetime(2023, 7, 13, 5, 10, tzinfo=datetime.timezone.utc),
        #     #         )
        #     #     ],
        #     # ),
        #     (
        #         "CA-QC",
        #         [
        #             (
        #                 datetime.datetime(2023, 7, 13, 5, 15, tzinfo=datetime.timezone.utc),
        #                 datetime.datetime(2023, 7, 14, 8, 0, tzinfo=datetime.timezone.utc),
        #             )
        #         ],
        #     )
        # ]

    def start_training(self):
        interval_idx = 0
        shutting = False
        while interval_idx < len(self.predicted_intervals) and not shutting:
            code, time_intervals = self.predicted_intervals[interval_idx]
            zone_idx = self.zone_indices[interval_idx]
            zone = codes_to_gcloud_zones[code]
            if zone != self.current_zone:
                print("Moving VM")
                start_moving_time = time.time()
                new_instance = google_cloud_move_vm(
                    current_zone=self.current_zone,
                    current_instance_name=self.current_instance_name,
                    project_id=self.project_id,
                    new_zone=zone,
                )
                print(
                    f"Moving completed in {int(time.time()-start_moving_time)} seconds"
                )
                self.current_vm_ip = (
                    new_instance.network_interfaces[0].access_configs[0].nat_i_p
                )
                self.current_zone = zone
                sleep(60)  # To let machine start

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
            if self.load_states:
                command_arguments += " --load_states"
            print("SSH Starting")
            channel, stdout = setup_ssh_execution(
                self.current_vm_ip,
                self.ssh_port,
                self.ssh_python_path,
                self.ssh_vm_main_path,
                username=self.ssh_username,
                command_arguments=command_arguments,
            )

            try:
                out = b''
                while True:
                    step = 500
                    out = out + stdout.read(step)
                    if out == b'':
                        break
                    try:
                        line = out.decode()
                        print(line, end='\r')
                        out = b''
                        if "Traceback" in line:
                            pass
                        if "End of training." in line:
                            print("Stopping master machine")
                            shutting = True
                            break
                    except UnicodeDecodeError:
                        out = out
            except KeyboardInterrupt:
                channel.send("\x03")
                for line in iter(stdout.readline, ""):
                    print(line, end="")
                print('Interrupted')
            
            interval_idx += 1
            if datetime.datetime.now() > self.last_prediction_time + datetime.timedelta(
                seconds=self.intervals_prediction_period
            ):
                self.last_prediction_time = datetime.datetime.now()
                co2_forecast = self.co2_predictor.predict_co2()
                self.predicted_intervals = self.interval_generator.generate_intervals(
                    forecasts=co2_forecast,
                    exclude_zones=self.exclude_zones,
                    include_zones=self.include_zones,
                    current_machine=zone_idx,
                )
                interval_idx = 0

            self.load_states = True

    def stop_training(self):
        stop_instance(
            project_id=self.project_id,
            zone=self.current_zone,
            instance_name=self.current_instance_name,
        )
