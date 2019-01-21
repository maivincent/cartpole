from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np




class ScoreLogger:

    def __init__(self, env_name, iteration):
        self.SCORES_CSV_PATH = "./scores/scores_" + str(iteration) + ".csv"
        self.MEANS_CSV_PATH = "./scores/means_" + str(iteration) + ".csv"
        self.SCORES_PNG_PATH = "./scores/scores_" + str(iteration) + ".png"
        self.SOLVED_CSV_PATH = "./scores/solved.csv"
        self.SOLVED_PNG_PATH = "./scores/solved.png"
        self.AVERAGE_SCORE_TO_SOLVE = 195
        self.CONSECUTIVE_RUNS_TO_SOLVE = 100

        self.scores = deque(maxlen=self.CONSECUTIVE_RUNS_TO_SOLVE)
        self.mean_scores = deque(maxlen=self.CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name
        self.iteration = iteration

        if os.path.exists(self.SCORES_PNG_PATH):
            os.remove(self.SCORES_PNG_PATH)
        if os.path.exists(self.SCORES_CSV_PATH):
            os.remove(self.SCORES_CSV_PATH)
        if os.path.exists(self.MEANS_CSV_PATH):
            os.remove(self.MEANS_CSV_PATH)

    def add_score(self, score, run):
        self.scores.append(score)
        if len(self.scores) == 1:
          mean_score = score
        else:
          mean_score = self.mean_scores[-1] + (score-self.mean_scores[-1])/len(self.scores)
        mean_score = mean(self.scores)
        self.mean_scores.append(mean_score)

        print("Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores)) + ")\n")

        self._save_csv(self.SCORES_CSV_PATH, score)
        self._save_csv(self.MEANS_CSV_PATH, mean_score)
        self._save_png(input_path=self.SCORES_CSV_PATH,
                       output_path=self.SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=self.CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True,
                       show_means=True,
                       means_input_path = self.MEANS_CSV_PATH)       

   
        ## Check if solved condition is achieved.
        if mean_score >= self.AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= self.CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run-self.CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(self.SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=self.SOLVED_CSV_PATH,
                           output_path=self.SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            return True

        return False

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend, show_means = False, means_input_path = None):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        if show_means:
            means = []
            means_x = []
            with open(means_input_path, "r") as scores:
                reader = csv.reader(scores)
                data = list(reader)
                for i in range(0, len(data)):
                    means_x.append(int(i))
                    means.append(float(data[i][0]))
            plt.plot(means_x, means, label="mean last 100")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [self.AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(self.AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])
