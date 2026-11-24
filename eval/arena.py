"""Seat-rotated evaluation. Usage: python -m eval.arena [n_games]"""
import sys, random, statistics as st
from wizard.engine import play_game
from wizard.agents import RandomAgent, HeuristicAgent


def evaluate(make_focal, make_opp, n_games=500, seed0=0):
    """Focal agent plays each seed in each of the 4 seats vs 3 opponents. Returns mean, 95% CI."""
    diffs = []
    for g in range(n_games):
        for seat in range(4):
            ag = [make_opp(g, i) for i in range(4)]
            ag[seat] = make_focal(g)
            tot = play_game(ag, seed0 + g)
            others = [tot[i] for i in range(4) if i != seat]
            diffs.append(tot[seat])
    m = st.mean(diffs)
    ci = 1.96 * st.stdev(diffs) / len(diffs) ** 0.5
    return m, ci


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rnd = lambda g, i=0: RandomAgent(random.Random(g * 10 + i))
    print("Heuristic vs 3 Random : mean score %.1f +- %.1f" % evaluate(lambda g: HeuristicAgent(), rnd, n))
    print("Random vs 3 Heuristic : mean score %.1f +- %.1f" % evaluate(lambda g: rnd(g, 9), lambda g, i: HeuristicAgent(), n))
    print("Heuristic vs 3 Heuristic: mean score %.1f +- %.1f" % evaluate(lambda g: HeuristicAgent(), lambda g, i: HeuristicAgent(), n))
