import random
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed


def fair_spin():
    """
    Simulates a fair spin of a six-possibility spinner. Relies on Python's baked-in RNG
    (which uses Mersenne Twister) to achieve a fair outcome.
    :return: integer from 1 to 6
    """
    return random.randint(1, 6)


def simulate_a_game():
    # A representation of all the 'values' of the squares of the board.
    # Positive numbers indicate squares that hold a ladder that promotes the
    # pawn N spots forward; negative numbers indicate squares that hold a chute that
    # demotes the pawn N spots backward.
    board = [
        37, 0, 0, 10, 0, 0, 0, 0, 22, 0,
        0, 0, 0, 0, 0, -10, 0, 0, 0, 0,
        21, 0, 0, 0, 0, 0, 0, 56, 0, 0,
        0, 0, 0, 0, 0, 8, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, -22, -38, 0,
        16, 0, 0, 0, 0, -3, 0, 0, 0, 0,
        0, -43, 0, -4, 0, 0, 0, 0, 0, 0,
        20, 0, 0, 0, 0, 0, 0, 0, 0, 20,
        0, 0, 0, 0, 0, 0, -63, 0, 0, 0,
        0, 0, -20, 0, -20, 0, 0, -20, 0, 999
    ]

    initial_squares = np.zeros(100, dtype='int')
    final_squares = np.zeros(100, dtype='int')

    # promotions = sum(map(lambda x: x > 0, board))
    # demotions = sum(map(lambda x: x < 0, board))

    done = False
    pos = -1
    spins = 0
    chutes = 0
    ladders = 0

    while not done:
        # Spin the 'spinner' and move our pawn that many spots forward
        spinval = fair_spin()
        pos += spinval

        # Don't run off the end of the board! Also, you have to spin
        # the exact number to land on the winning square
        if pos > len(board) - 1:
            pos -= spinval  # nope, no move

        # Count the spins and note the spot for heatmap later
        spins += 1

        initial_squares[pos] += 1

        # Now obey the chute or ladder (demotion or promotion) that is at that spot, or note a win
        reposition = board[pos]
        if reposition == 999:
            done = True
        else:
            pos += reposition
            if reposition < 0:
                chutes += 1
            elif reposition > 0:
                ladders += 1

        final_squares[pos] += 1

    return spins, initial_squares, final_squares, chutes, ladders


if __name__ == "__main__":
    spincount_by_trial = []
    chutecount_by_trial = []
    laddercount_by_trial = []

    landed_squares = np.zeros(100, dtype=int)
    final_squares = np.zeros(100, dtype=int)

    random.seed()
    trials = 500000
    do_plot = True
    do_parallel = True  # set to False for debugging (debugger barfs on parallel code)

    start = timer()
    if do_parallel:
        parallel_results = Parallel(n_jobs=8)(delayed(
            simulate_a_game)() for _ in range(trials)  # <-- this is what will be parallelized
                                              )
        # Joblib doco shows how to deal with functions that return multiple values like perform_trial() does.
        # See https://joblib.readthedocs.io/en/latest/parallel.html
        # (zip with * unzips a list)
        spincount_by_trial, initial_squares_by_trial, final_squares_by_trial, chutecount_by_trial, laddercount_by_trial = zip(*parallel_results)
        for v in initial_squares_by_trial:
            landed_squares += v
        for v in final_squares_by_trial:
            final_squares += v

    else:
        for _ in range(trials):
            spin_count, initial_squares_for_trial, final_squares_for_trial, chutes_hit, ladders_hit = simulate_a_game()
            spincount_by_trial.append(spin_count)
            chutecount_by_trial.append(chutes_hit)
            laddercount_by_trial.append(ladders_hit)
            landed_squares += initial_squares_for_trial
            final_squares += final_squares_for_trial

    end = timer()
    print(f"{end - start} seconds elapsed time")  # Time in seconds, e.g. 5.38091952400282

    temp = np.array(spincount_by_trial)
    print(f"Minimum number of spins to win: {min(temp)}")
    print(f"Maximum number of spins to win: {max(temp)}")
    print(f"Arithmetic mean number of ladders hit by player: {round(np.average(laddercount_by_trial))}")
    print(f"Arithmetic mean number of chutes hit by player: {round(np.average(chutecount_by_trial))}")
    print(f"Arithmetic mean of spins to win: {np.mean(temp)}")
    print(f"Median of spins to win: {np.median(temp)}")
    print(f"Standard deviation = {np.std(temp)}")

    if do_plot:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 25))
        plt.title(f"Results for {trials} Trials")

        ##########################
        # Row 0
        axes[0][0].title.set_text('Cumulative Distribution of Number-of-Spins To Win Game')
        axes[0][0].set_xlabel('Spins to win')
        s = pd.Series(spincount_by_trial)
        s.plot.hist(ax=axes[0][0], grid=True, bins=range(0, 200, 10), rwidth=0.95, color='#607c8e', cumulative=True)
        axes[0][0].set(xticks=range(0, 200, 10))

        axes[0][1].title.set_text('Distribution of Number-of-Spins To Win Game')
        axes[0][1].set_xlabel('Spins to win')
        s.plot.hist(ax=axes[0][1], grid=True, bins=range(0, 200, 10), rwidth=0.95, color='#607c8e', cumulative=False)
        axes[0][1].set(xticks=range(0, 200, 10))

        ##########################
        # Row 1
        axes[1][0].title.set_text('Board Heatmap of Initial Landing Positions')
        axes[1][0].axes.xaxis.set_visible(False)
        axes[1][0].axes.yaxis.set_visible(False)

        # Reshape the spot-usage bins to a 10x10 matrix. Flip it around the 0 axis
        # to make it look like the board (bottom-to-top)...
        t = landed_squares.reshape((10, 10))
        t = np.flip(t, axis=0)

        # ...convert to a Pandas dataframe for purposes of reversing every other row
        # to make it comparable with the physical board. See
        # https://stackoverflow.com/questions/37280681/reverse-even-rows-in-a-numpy-array-or-pandas-dataframe
        df = pd.DataFrame(t)
        df.iloc[0::2, :] = df.iloc[0::2, ::-1].values
        #       ^  ^                        ^
        #       |  |                  Reverse
        #  Start   |
        # Every other row
        sns.heatmap(df, annot=True, fmt='d', cmap="coolwarm", ax=axes[1][0])

        axes[1][1].title.set_text('Board Heatmap of End-of-turn Positions')
        axes[1][1].axes.xaxis.set_visible(False)
        axes[1][1].axes.yaxis.set_visible(False)

        # Reshape the spot-usage bins to a 10x10 matrix. Flip it around the 0 axis
        # to make it look like the board (bottom-to-top)...
        t = final_squares.reshape((10, 10))
        t = np.flip(t, axis=0)

        # ...convert to a Pandas dataframe for purposes of reversing every other row
        # to make it comparable with the physical board. See
        # https://stackoverflow.com/questions/37280681/reverse-even-rows-in-a-numpy-array-or-pandas-dataframe
        df = pd.DataFrame(t)
        df.iloc[0::2, :] = df.iloc[0::2, ::-1].values
        #       ^  ^                        ^
        #       |  |                  Reverse
        #  Start   |
        # Every other row
        sns.heatmap(df, annot=True, fmt='d', cmap="coolwarm", ax=axes[1][1])

        ##########################
        # Row 2
        axes[2][0].title.set_text('Distribution of Number-of-Ladders-Hit In A Game')
        axes[2][0].set_xlabel('Ladders hit')
        s = pd.Series(laddercount_by_trial)
        s.plot.hist(ax=axes[2][0], grid=True, bins=range(0, 25, 1), rwidth=0.95, color='#607c8e', cumulative=False)
        axes[2][0].set(xticks=range(0, 25, 1))

        axes[2][1].title.set_text('Distribution of Number-of-Chutes-Hit In A Game')
        axes[2][1].set_xlabel('Chutes hit')
        s = pd.Series(chutecount_by_trial)
        s.plot.hist(ax=axes[2][1], grid=True, bins=range(0, 25, 1), rwidth=0.95, color='#607c8e', cumulative=False)
        axes[2][1].set(xticks=range(0, 25, 1))

        plt.show()

    print("Done!")
