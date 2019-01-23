from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np

NUMBER_RUNS = 17

class ScoreAnalyst:
    def __init__(self, nb_runs):
        self.nb_runs = nb_runs
        self.score_csv_root_path = "scores_"
        self.score_csv_ext_path = ".csv"
        self.scores_list = [[] for k in range(self.nb_runs)]
        self.scores_lengths = [[] for k in range(self.nb_runs)]
        self.scores_average_step = []
        
        self.max_length = 0
        self.min_length = 0


    def make_score_analysis(self):
        for i in range(self.nb_runs):
            self.get_score_data(i+1)

        self.max_length = max(self.scores_lengths)
        self.min_length = min(self.scores_lengths)

        self.complete_score_datas()
        self.scores_average_step = [[] for k in range(self.max_length)]


        for j in range (self.max_length):
            sum_at_step = 0
            for i in range(self.nb_runs):
                sum_at_step += self.scores_list[i][j]
            self.scores_average_step[j] = sum_at_step/self.nb_runs

        avg_score_step_plot_title = "Average score per step over " + str(self.nb_runs) + " runs"
        avg_score_step_x_label = "Step"
        avg_score_step_y_label = "Average score"
        avg_score_step_output_path = "./average_score_step.png"

        self.save_list_png(avg_score_step_output_path, self.scores_average_step, avg_score_step_x_label, avg_score_step_y_label, avg_score_step_plot_title)

    def get_score_data(self, it):
        # Open score file and put it into the scores_list
        file_name = self.score_csv_root_path + str(it) + self.score_csv_ext_path
        with open(file_name, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            self.scores_lengths[it-1] = len(data)
            for i in range(0, len(data)):
                self.scores_list[it-1].append(int(data[i][0]))

    def complete_score_datas(self):
        # Complete score datas until they reach max length with the mean of the last 25
        for i in range(len(self.scores_list)):
            while len(self.scores_list[i]) < self.max_length:
                avg_last_25 = sum(self.scores_list[i][-25:])/25
                self.scores_list[i].append(avg_last_25)


    def save_list_png(self, output_path, y, x_label, y_label, plot_title):
        x = range(len(y))

        plt.subplots()
        plt.plot(x, y, label="average score per step")
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
       
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()








if __name__ == "__main__":
    score_analyst = ScoreAnalyst(NUMBER_RUNS)
    score_analyst.make_score_analysis()
