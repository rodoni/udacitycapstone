import pandas as pd
import numpy as np

FILE_PATH="/home/rodoni/DataSet/Amazon/train.csv"

train_file = pd.read_csv(FILE_PATH)

number_of_rows = (train_file.shape[0])
index = 0
activit = ["none", "primary", "water", "agriculture", "road", "bare_ground", "artisinal_mine","conventional_mine",
           "blooming", "habitation", "selective_logging", "cultivation", "slash_burn"   ]
weather_data = pd.DataFrame(index=range(number_of_rows), columns=["image_name", "weather_condition"])
activit_data = pd.DataFrame(index=range(number_of_rows), columns=)

while index < number_of_rows:
    name = train_file.get_value(index, "image_name")
    tags = train_file.get_value(index, "tags")

    if "clear" in tags:

    index = index + 1
