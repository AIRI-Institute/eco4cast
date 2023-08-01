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
import pandas as pd


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
        print('Gathering info and predicting CO2 ... This may take a while ...')
        self.co2_forecast = self.co2_predictor.predict_co2()
        print('CO2 predicted')
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
            printed = False
            while datetime.datetime.now(datetime.timezone.utc) < time_intervals[0][
                0
            ] - datetime.timedelta(seconds=30):
                if not printed:
                    print(f'Waiting till next interval start {time_intervals[0][0].strftime("%Y%m%d%H%M%S")}')
                time.sleep(1)

            argument_intervals = [
                (s.strftime("%Y%m%d%H%M%S"), e.strftime("%Y%m%d%H%M%S"))
                for s, e, _ in time_intervals
            ]
            
            argument_intervals = [item for t in argument_intervals for item in t]
            argument_intervals = ",".join(argument_intervals)

    
            command_arguments = f"--times {argument_intervals}"
            if self.load_states:
                command_arguments += " --load_states"

            co2_means = [str(co2) for _, _, co2 in time_intervals]
            command_arguments += f' --co2_means {",".join(co2_means)}'

            print("SSH Starting")
            channel, stdout = self.__setup_ssh_execution(
                self.current_vm_ip,
                self.ssh_port,
                self.ssh_python_path,
                self.ssh_vm_main_path,
                username=self.ssh_username,
                command_arguments=command_arguments,
            )
            try:
                out = b""
                while True:
                    step = 500
                    out = out + stdout.read(step)
                    if out == b"":
                        break
                    try:
                        line = out.decode()
                        print(line, end="\r")
                        out = b""
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
                print("Interrupted")
                shutting = True

            if not shutting:
                interval_idx += 1
                if datetime.datetime.now() > self.last_prediction_time + datetime.timedelta(
                    seconds=self.intervals_prediction_period
                ):
                    self.last_prediction_time = datetime.datetime.now()
                    co2_forecast = self.co2_predictor.predict_co2()
                    self.predicted_intervals = self.interval_generator.generate_intervals(
                        forecasts=co2_forecast,
                        current_machine=zone_idx,
                    )
                    interval_idx = 0
                self.load_states = True
        
        self.__calc_emission()

    def __calc_emission(self):
        sftp = self.ssh_client.open_sftp()
        path = "/".join(self.ssh_vm_main_path.split("/")[:-1]) + "/emission.csv"
        sftp.get(path, "emission.csv")
        emission_df = pd.read_csv("emission.csv")
        total_emission = emission_df['CO2_emissions(kg)'].sum()*1000
        total_electricity = emission_df['power_consumption(kWh)'].sum()

        co2_average_intensity = self.co2_forecast.mean()
        co2_average_g = (total_electricity * co2_average_intensity)
        delta = total_emission / co2_average_g
        if delta <= 1:
            print(f'Your total CO2 emissions are {total_emission:.6f} g and this is {(1-delta)*100:.2f}% less than average emission ({co2_average_g:.6f} g)')
        else:
            print(f'Your total CO2 emissions are {total_emission:.6f} g and this is {(delta-1)*100:.2f}% higher than average emission ({co2_average_g:.6f} g)')

    def __setup_ssh_execution(
        self,
        ip,
        ssh_port,
        python_path,
        vm_main_path,
        username="scheduler",
        command=None,
        command_arguments="",
    ):
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(
            hostname=ip, port=ssh_port, username=username, timeout=10
        )

        vm_main_folder = "/".join(vm_main_path.split("/")[:-1])
        vm_main_file = vm_main_path.split("/")[-1]

        command = f"cd {vm_main_folder}; {python_path} {vm_main_file}"
        command += " " + command_arguments

        transport = self.ssh_client.get_transport()
        bufsize = -1
        channel = transport.open_session()
        channel.get_pty()
        channel.exec_command(command)
        # stdin = channel.makefile_stdin("wb", bufsize)
        stdout = channel.makefile("r", bufsize)
        # stderr = channel.makefile_stderr("r", bufsize)

        return channel, stdout

    def stop_training(self):
        stop_instance(
            project_id=self.project_id,
            zone=self.current_zone,
            instance_name=self.current_instance_name,
        )
