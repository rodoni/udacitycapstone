import pandas as pd


class InputFileReader(object):

    def __init__(self, path):
        self.path = path

    def get_data(self):

        train_file = pd.read_csv(self.path)
        number_of_rows = (train_file.shape[0])
        index = 0
        weather_data = pd.DataFrame(index=range(number_of_rows), columns=["image_name", "weather_condition"])
        activit_data = pd.DataFrame(index=range(number_of_rows),
                            columns=["image_number", "primary", "water", "agriculture", "road", "bare_ground",
                                     "artisinal_mine", "conventional_mine", "blooming", "habitation",
                                     "selective_logging", "cultivation", "slash_burn"])
        activit_data = activit_data.fillna(0)

        # values for weather situation
        #  clear : 0
        #  cloudy: 1
        #  haze  : 2
        #  partly_cloudy : 3
        #

        while index < number_of_rows:
            name = train_file.get_value(index, "image_name")
            tags = train_file.get_value(index, "tags")

            weather_data.set_value(index, "image_name", index)

            if "clear" in tags:
                weather_data.set_value(index, "weather_condition", 0)
            if "cloudy" in tags:
                weather_data.set_value(index,"weather_condition", 1)
            if "haze" in tags:
                weather_data.set_value(index,"weather_condition", 2)
            if "partly_cloudy" in tags:
                weather_data.set_value(index,"weather_condition", 3)
            index = index + 1

        index = 0

        while index < number_of_rows:

            name = train_file.get_value(index, "image_name")
            tags = train_file.get_value(index, "tags")

            activit_data.set_value(index, "image_number", index)

            if "primary" in tags:
                activit_data.set_value(index, "primary", 1)
            if "water" in tags:
                activit_data.set_value(index, "water", 1)
            if "agriculture" in tags:
                activit_data.set_value(index, "agriculture", 1)
            if "road" in tags:
                activit_data.set_value(index, "road", 1)
            if "bare_ground" in tags:
                activit_data.set_value(index, "bare_ground", 1)
            if "artisinal_mine" in tags:
                activit_data.set_value(index, "artisinal_mine", 1)
            if "conventional_mine" in tags:
                activit_data.set_value(index, "conventional_mine", 1)
            if "blooming" in tags:
                activit_data.set_value(index, "blooming", 1)
            if "habitation" in tags:
                activit_data.set_value(index, "habitation", 1)
            if "selective_logging" in tags:
                activit_data.set_value(index, "selective_logging", 1)
            if "cultivation" in tags:
                activit_data.set_value(index, "cultivation", 1)
            if "slash_burn" in tags:
                activit_data.set_value(index, "slash_burn", 1)

            index = index + 1

        return activit_data, weather_data, index







