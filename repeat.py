from cartpole import cartpole


REPETITIONS = 200

class Params:
    def __init__(self):
        self.ENV_NAME = "CartPole-v1"
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.001

        self.MEMORY_SIZE = 1000000
        self.BATCH_SIZE = 20

        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.995


def repeat():
    params = Params()
    for i in range(0, REPETITIONS):
        print("[repeat] Starting repetition " + str(i) + "... \n")
        cartpole(i, params)
        print("-------------------------------------------------- \n")




if __name__ == "__main__":
    repeat()
