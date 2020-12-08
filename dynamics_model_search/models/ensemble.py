from models.dynamics_model import DynamicsModel
import torch, random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DummyModel:
    def __init__(self, parent):
        self.parent = parent

    def train(self):
        for model in self.parent.ensemble:
            model.model.train()

    def eval(self):
        for model in self.parent.ensemble:
            model.model.eval()

class Ensemble(DynamicsModel):
    def __init__(self, model, env_in, size=4, *kwargs):
        DynamicsModel.__init__(self, env_in)
        self.ensemble_size = size
        self.ensemble = [model(env_in, *kwargs) for _ in range(self.ensemble_size)]
        self.model = DummyModel(self)
        self.device = self.ensemble[0].device

    def exec_reset(self, obs_in, h_split):
        reset_out = []
        for i in range(self.ensemble_size):
            _, model_h = self.ensemble[i].reset(obs_in, h_split[i])
            reset_out.append(model_h)
        return reset_out

    def reset(self, obs_in, h=None):
        if h is not None:
            h_split = h.chunk(self.ensemble_size, -1)
        else:
            h_split = [None]*self.ensemble_size
        reset_out = self.exec_reset(obs_in, h_split)
        return obs_in, torch.cat(reset_out, -1)

    def process_step(self, steps, choice, certainty=False, state=False):
        new_obs = []
        state_out = []
        uncertainties = []

        for step in steps:
            state_out_step =    None
            unc_step =          None

            if state and certainty:     new_obs_step, state_out_step, unc_step = step
            elif state:                 new_obs_step, state_out_step = step
            elif certainty:             new_obs_step, unc_step = step
            else:                       new_obs_step = step

            new_obs.append(new_obs_step.unsqueeze(0))
            state_out.append(state_out_step)
            uncertainties.append(unc_step)

        new_obs = torch.cat(new_obs)
        new_obs_mean = torch.mean(new_obs, 0)
        new_obs_std = torch.std(new_obs, 0)
        new_obs_sample = torch.distributions.Normal(new_obs_mean, new_obs_std).sample()

        if state:       state_out = torch.cat(state_out, -1)
        # if certainty:   unc = unc[choice]
        # new_obs = new_obs[choice]

        if certainty:   uncertainty = new_obs_std
        new_obs = new_obs_sample

        if state and certainty:     return new_obs, state_out, uncertainty
        if state:                   return new_obs, state_out
        if certainty:               return new_obs, uncertainty
        else:                       return new_obs

    def exec_step(self, action_in, obs_in, save, state, state_in, certainty):
        steps = []
        for i in range(self.ensemble_size):
            model = self.ensemble[i]
            step = model.step(action_in, obs_in[i], save, state, state_in, certainty)
            steps.append(step)
        return steps

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None, certainty=False):
        choice = random.randint(0, self.ensemble_size-1)

        if obs_in is not None and state_in:
            b = len(obs_in[1])
            state_var = torch.cat(obs_in[1]).chunk(self.ensemble_size, -1)
            obs_var = obs_in[0]
            obs_in = [(obs_var, state_var[i].chunk(b)) for i in range(self.ensemble_size)]

        steps = self.exec_step(action_in, obs_in, save, state, state_in, certainty)

        return self.process_step(steps, choice, certainty, state)

    def chunk(self, list, n):
        l = int(len(list)/n)
        return [list[l*i:l*(i+1)] for i in range(n)]

    def update(self, data):
        data = self.chunk(data, self.ensemble_size)
        losses = []
        for i in range(self.ensemble_size):
            loss = self.ensemble[i].update(data[i])
            losses.append(loss)
        return losses

    def save(self, save_str):
        for i in range(self.ensemble_size):
            self.ensemble[i].save(save_str+'_model_'+str(i))

    def load(self, save_str):
        for i in range(self.ensemble_size):
            self.ensemble[i].load(save_str+'_model_'+str(i))

    def save_dict(self):
        return [model.save_dict() for model in self.ensemble]

    def load_dict(self, d):
        for i in range(self.ensemble_size):
            self.ensemble[i].load_dict(d[i])

    def to(self, in_device):
        for model in self.ensemble:
            model.to(in_device)
