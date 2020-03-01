"""
Code modified from: https://github.com/alshedivat/lola/tree/master/lola
"""
import numpy as np
from gym.envs.multi_agent.matrix_social_dilemma import MatrixSocialDilemma

def test_MatrixSocialDilemma():
    n_test_games=20
    n_test_step=5
    # Play n games
    for i in range(n_test_games):
        payout_matrix = np.random.randint(-10, 10, (2,2))
        social_dilemma = MatrixSocialDilemma(payout_matrix=payout_matrix)
        o = social_dilemma.reset()

        for agent_num in range(len(o)):
            assert o[agent_num] == (social_dilemma.NUM_STATES -1)

        # Play n  steps
        for n in range(n_test_step):
            action = np.random.randint(0, 2, (2,)).tolist()
            o, r, done, info = social_dilemma.step(action=action)

            # Assume 2 agents
            for agent_num in range(len(r)):
                current_agent_a = action[agent_num]
                other_agent_a = action[(agent_num +1 ) % len(r)]
                assert (r[agent_num] ==
                       social_dilemma.payout_mat[current_agent_a][other_agent_a])
