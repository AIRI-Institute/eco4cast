from eco4cast.master_machine.interval_predictor import (
    CO2Predictor,
    IntervalGenerator,
)
import os
from pathlib import Path
from eco4cast.master_machine.master_machine_controller import Controller


username = "scheduler"
electricity_maps_api_key = "gUZ7QzkmU91g50yliykpgOUdGQGUIDcD" # your API key here
controller = Controller(
    credentials_path=Path(os.getcwd()) / "application_default_credentials.json",
    ssh_username=username,
    ssh_python_path=f"/home/{username}/venv/bin/python",
    ssh_vm_main_path=f"/home/{username}/scheduler_task/vm_main.py",
    current_vm_ip="35.203.71.83",
    ssh_port=22,
    current_zone="northamerica-northeast1-c",
    current_instance_name="instance-3",
    project_id="test-smart-scheduler",
    intervals_prediction_period=3600,  # seconds
    co2_predictor=CO2Predictor(electricity_maps_api_key),
    interval_generator=IntervalGenerator(),
)


controller.start_training()
print("Successfully finished all intervals. Stopping Instance")
controller.stop_training()
