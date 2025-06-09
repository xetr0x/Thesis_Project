from sklearn.ensemble import IsolationForest
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)  # changes CWD to the root

sys.path.append(project_root)

print("Current Working Directory:", os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functions.global_functions as gb
from sklearn.preprocessing import StandardScaler
import copy
import joblib


DEFAULT_WINDOW_SIZE = 5000   # size of sliding window
DEFAULT_STEP_SIZE = 500  # amount of indexes of the sliding window it will move
ANOMALY_THRESHOLD = 0.10 # retrain if more than 10% is anomalies.
SEC_SIZE = 500 # the size of the "section"
RESULT_FILE = 'iso_results.csv' # file where results are saved
FEATURES = ['Temperature (°C)', 'Humidity (%)',
            'Raw VOC', 'IR Light', 'Visible Light', 'CO2 (ppm)']

class Iso_Forest:

#TODO:  first time u append isolation forest use own df then from there u go

# Sliding window makes it instead adaptable
    def __init__(self, number_of_trees=100,
                 contamination=0.008, sample_size=128,
                 reading_types=FEATURES,
                 window_size = DEFAULT_WINDOW_SIZE,
                 random_state=42):

        self.n_trees = number_of_trees
        self.contamination = contamination
        self.sample_size = sample_size
        self.features = reading_types
        self.curr_index = 0

        data = gb.collect_data(0, DEFAULT_WINDOW_SIZE)
        df = gb.data_cleaner(data, features=self.features)


# keeps track of section samples, section is finnished, it gets
# evoluated if model needs fitting, added to csv file and reset for
# the next batch stream
        self.df_last_sec = []

        df, self.current_scaler = self.fix_scaler(df[self.features])
        self.model = IsolationForest(n_estimators=self.n_trees,
                                     contamination=self.contamination,
                                     max_samples=self.sample_size)
        self.model.fit(df[self.features])



    def fix_scaler(self, train_x):
        scaler = StandardScaler()

        scaler.fit(train_x[FEATURES])
        x_scaled = scaler.transform(train_x[FEATURES])
        x_scaled = pd.DataFrame(x_scaled, columns=FEATURES)
        return x_scaled, scaler


# adaptive sliding window
    def sliding_window_test(self):
        copied_section = self.df_last_sec.copy()
        section = pd.concat(copied_section, ignore_index=True)
        scaled_section = self.current_scaler.transform(section[FEATURES])
        df_section = pd.DataFrame(scaled_section, columns=FEATURES)
        test = self.model.predict(df_section[FEATURES])


        mean_control = np.mean(test == -1)



        if (mean_control > ANOMALY_THRESHOLD):

            remember_index = self.curr_index
            current_model = copy.deepcopy(self.model)

            while (True):
                try:
                    self.curr_index = self.curr_index+DEFAULT_STEP_SIZE+DEFAULT_STEP_SIZE
                    data = gb.data_cleaner(dataframe_toclean=gb.collect_data(
                        self.curr_index, DEFAULT_WINDOW_SIZE+DEFAULT_STEP_SIZE),
                        features=self.features)
                    
                    if (len(data) != DEFAULT_WINDOW_SIZE):
                        self.curr_index = remember_index # Goes back to the last known good index
                        self.model = copy.deepcopy(current_model) # returns to previous state
                        print("end of training data, might be anomalous data")
                        return False

                    data, scaler = self.fix_scaler(data[self.features])
                    self.model.fit(data[self.features])
                    copied_section = self.df_last_sec.copy()
                    section = pd.concat(copied_section, ignore_index=True)
                    scaled_last_sec = scaler.transform(section[FEATURES])
                    df_section_scaled = pd.DataFrame(scaled_last_sec, columns=FEATURES)


                    test = self.model.predict(df_section_scaled[FEATURES])
                    mean_control = np.mean(test == -1)

                    if (mean_control < ANOMALY_THRESHOLD):
                        print("ISOASD ADAPTED!")
                        self.current_scaler = scaler # new scaler bc new fit
                        return True


                except (IndexError, ValueError) as e:
                    print("Section anomalous!")
                    print("error: ", e)
                    self.curr_index = remember_index # Goes back to the last known good index
                    self.model = copy.deepcopy(current_model) # returns to previous state
                    return False




# This function is made for a async situation
    def predict(self, sample):
        sample_copy = sample.copy()
        self.df_last_sec.append(sample_copy)


        scaled_sample = self.current_scaler.transform(sample_copy[FEATURES])
        scaled_sample = pd.DataFrame(scaled_sample, columns=FEATURES)
        score = self.model.decision_function(scaled_sample[FEATURES])
        sample_score = self.model.score_samples(scaled_sample[FEATURES])
        anomaly = self.model.predict(scaled_sample[FEATURES])



        if (len(self.df_last_sec) >= SEC_SIZE):
            # if feedback = false: section is anomalous, if true it is not and it was successfull. (for future work)
            feedback = self.sliding_window_test()
            # later send feedback as a signal to the server.
            self.data_to_memory()  # for sending the df section to csv
            self.df_last_sec = [] # resets the section

        return score, anomaly, sample_score


# saves the data to memory
    def data_to_memory(self):
        section = pd.concat(self.df_last_sec, ignore_index=True)
        # section.to_csv('Resources/Our_Dataset/Iso_data.csv', mode='a', header=False, index=False, sep=';')




# For later, to make it optimal for future
# deletes data after a certain size
    def automatic_data_adaptation():
        pass

# This is simply for testing if it works correctly
# and for fixing contamination level
    def initial_test(self, data_frame):
        df_copy = data_frame.copy()
        anomalies = 0
        non_anomalies = 0
        df_scaled = self.current_scaler.transform(df_copy[FEATURES])
        scaled_sample = pd.DataFrame(df_scaled, columns=FEATURES)
        df_copy['score'] = self.model.decision_function(scaled_sample[FEATURES])
        df_copy['anomaly'] = self.model.predict(scaled_sample[FEATURES])

        print("running test, with size: ", DEFAULT_STEP_SIZE )
        df_copy.to_csv(RESULT_FILE, mode='a', header=True, index=False, sep=';')
        
        gb.plot_graph(df_copy, x_axis_label="instance",
                  y_axis_label='temperature',
                      title='anomaly score', x_var='Timestamp',
                      y_var='Temperature (°C)')

        gb.plot_graph(df_copy, x_axis_label="instance",
                  y_axis_label='Humidity (%)',
                      title='anomaly score', x_var='Timestamp',
                      y_var='Humidity (%)')
        gb.plot_graph(df_copy, x_axis_label="instance",
                  y_axis_label='Raw VOC',
                      title='anomaly score', x_var='Timestamp',
                      y_var='Raw VOC')
        gb.plot_graph(df_copy, x_axis_label="instance",
                  y_axis_label='IR Light',
                      title='anomaly score', x_var='Timestamp',
                      y_var='IR Light')
        gb.plot_graph(df_copy, x_axis_label="instance",
                  y_axis_label='Visible Light',
                      title='anomaly score', x_var='Timestamp',
                      y_var='Visible Light')
        gb.plot_graph(df_copy, x_axis_label="instance",
                  y_axis_label='CO2 (ppm)',
                      title='anomaly score', x_var='Timestamp',
                      y_var='CO2 (ppm)')
        plt.legend()
        plt.show()
        return data_frame





def main():
    model = Iso_Forest()
    model.initial_test(gb.collect_test_data())

if __name__ == "__main__":
    main()

