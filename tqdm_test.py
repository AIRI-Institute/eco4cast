from utils import CustomStartProgressBar
from time import sleep

progress_bar_1 = CustomStartProgressBar(100, description='Train')


for i in range(100):
    progress_bar_1()
    sleep(0.1)
    if i % 20 == 0:
        progress_bar_2 = CustomStartProgressBar(100, description='Val')
        for j in range(100):
            progress_bar_2()
            sleep(0.1)