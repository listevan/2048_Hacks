import random
import math
import torch
import numpy as np
from Game_Sim.game import Game


def select_action(
    state,
    model,
    steps_done,
    EPS_START=0.9,
    EPS_END=0.05,
    EPS_DECAY=5000,
    EX_MODEL=False,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # input = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0) # nn
            if not EX_MODEL:
                input = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(
                    0
                )  # cnn
            else:
                input = (
                    torch.tensor(state, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # 3d cnn

            output = torch.argsort(
                model(input).detach().cpu()
            ).squeeze()  # normally is selected via policy_net
            for action in output:
                g = Game()
                g.from_board(state, multidimensional=True)
                success, _, _ = g.move(action)
                if not success:
                    continue
                else:
                    return action, steps_done
            return np.random.choice(4), steps_done  # no available moves left
    else:
        return np.random.choice(4), steps_done
