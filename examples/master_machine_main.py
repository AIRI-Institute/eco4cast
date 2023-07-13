from smartscheduler.master_machine.interval_predictor import (
    CO2Predictor,
    IntervalGenerator,
)
from smartscheduler.master_machine.utils import code_names
import os
from pathlib import Path
from smartscheduler.master_machine.master_machine_controller import Controller


username = "scheduler"
controller = Controller(
    credentials_path=Path(os.getcwd()) / "application_default_credentials.json",
    ssh_username="scheduler",
    ssh_python_path=f"/home/{username}/venv/bin/python",
    ssh_vm_main_path=f"/home/{username}/scheduler_task/vm_main.py",
    current_vm_ip="35.203.81.70",
    ssh_port=22,
    current_zone="northamerica-northeast1-b",
    current_instance_name="instance-1",
    project_id="test-smart-scheduler",
    intervals_prediction_period=3600,  # seconds
    co2_predictor=CO2Predictor(),
    interval_generator=IntervalGenerator(
        code_names,
    ),
)


controller.start_training()
print("Successfully finished all intervals. Stopping Instance")
controller.stop_training()
