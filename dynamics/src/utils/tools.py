import numpy as np
import gym
from pytorch_src.trainers import train_walker as active_testing
from pytorch_src.trainers import train_walker as testing
from pytorch_src.env_learners.dnn_env_learner import DNNEnvLearner
from pytorch_src.env_learners.seq_env_learner import SeqEnvLearner
# from pytorch_src.env_learners.mdrnn_env_learner import MDRNNEnvLearner
from pytorch_src.env_learners.preco_gen_env_learner import PreCoGenEnvLearner
from pytorch_src.env_learners.preco_gen_attn_env_learner import PreCoGenAttnEnvLearner
from pytorch_src.env_learners.preco_env_learner import PreCoEnvLearner
from pytorch_src.env_learners.preco_attn_env_learner import PreCoAttnEnvLearner
from pytorch_src.env_learners.preco_gan_env_learner import PreCoGANEnvLearner
from pytorch_src.env_learners.multi_model_env_learner import MultiModelEnvLearner
from pytorch_src.env_learners.da_rnn_env_learner import DARNNEnvLearner
from envs.pybullet_wrappers import StateWrapper, RealerWalkerWrapper


import pybullet, pybullet_envs
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pybullet.connect(pybullet.DIRECT)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 100
# torch.manual_seed(seed)
# if device.type == 'cuda':
#     torch.cuda.manual_seed(seed)


def step_diff(a, b):
    if len(b[0].shape) > 1:
        diffs = np.zeros(b[0].shape[0])
    else:
        diffs = 0
    for i in range(len(b)):
        diff = np.square(b[i] - a[i][3])
        diffs += np.mean(diff)
    diffs = diffs/len(b)
    # avg_diff = np.mean(diffs, 1)
    return diffs

def test(model_dir='../models/'):
    # import os
    # print(os.listdir())
    nb_valid_episodes = 20
    i = 0
    episode_duration = 100
    episode_step = 0
    episode_reward = 0.
    max_ep_rew = -1000
    valid = []

    # env = testing.AntWrapper(gym.make("AntBulletEnv-v0"))
    # env = active_testing.RealerAntWrapper(gym.make("AntBulletEnv-v0"))
    env = RealerWalkerWrapper(gym.make("AntBulletEnv-v0"))

    load = '2019-12-01-15:53:51.pt' # Precogen 256 with replay and norm on entire replay (1M examples)

    load = model_dir+load

    lookahead = 10
    # env_learner = PreCoEnvLearner(env)
    # env_learner = PreCoAttnEnvLearner(env)
    # env_learner = PreCoGANEnvLearner(env)
    env_learner = PreCoGenEnvLearner(env)
    # env_learner = PreCoGenAttnEnvLearner(env)
    # env_learner = DARNNEnvLearner(env)
    # env_learner = DNNEnvLearner(env)
    # env_learner = MultiModelEnvLearner(env, PreCoGenEnvLearner, mode='test')
    # env_learner = SeqEnvLearner(env)
    # env_learner = MDRNNEnvLearner(env)

    ## TODO: Doesn't seem to give a consistent validation error when compared with the test_plan validation seems the model may not be loading correctly

    if load is not None:
        env_learner.load(load)
        # env_learner.model.load_state_dict(torch.load(load))
        # # env_learner.model = torch.load(load)
        # print(env_learner.model._modules['fc_out']._parameters['weight'])
        # env_learner.model.eval()
        # env_learner.generator = torch.load(load)

    # import pickle
    # valid = pickle.load(open('valid.pkl', 'r'))
    # print('Valid Loaded')
    # corr, single, seq = env_learner.get_loss(valid)
    # print('Valid Single: ' + str(single))
    # print('Valid Seq: ' + str(seq))
    # print('Valid Corr: ' + str(corr))
    # exit(0)
    chart = True
    # chart = False


    obs = env.reset()
    env_learner.reset(obs)
    open2_obs = obs
    open_obs = obs
    # closed_obs = obs

    closed = []
    open2 = []
    open = []
    real = []
    none = []

    episode_step = 0
    episode_reward = 0.
    max_ep_rew = -1000


    valid = []
    all_closed_diffs = []
    all_open_diffs = []
    final_open_drifts = []

    all_diffs = []

    all_valid = []
    path_uncertainty = []

    max_action = env.action_space.high
    while i < nb_valid_episodes:
        # action = np.random.uniform(-1, 1, env.action_space.shape[0])
        action = np.clip(np.random.normal(0, 1, env.action_space.shape[0]), -1, 1)
        new_obs, r, done, info = env.step(max_action * action)
        if episode_duration > 0:
            done = (done or (episode_step >= episode_duration))
        valid.append([obs, max_action * action, r, new_obs, done, episode_step])
        all_valid.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1

        # closed_obs = env_learner.step(obs_in=obs, action_in=max_action * action, save=False)
        # open2_obs = env_learner.step(obs_in=open2_obs, action_in=max_action * action, episode_step=episode_step, save=True)
        if len(valid)%lookahead == 0 and lookahead > 0:
            open_obs = env_learner.step(obs_in=obs, action_in=max_action * action, save=True, state=False, state_in=False)
            # env_learner.obs_in = obs
        else:
            open_obs = env_learner.step(action_in=max_action * action, save=True, state=False)
        closed_obs = np.zeros_like(open_obs)
        open2_obs = np.zeros_like(open_obs)
        # open_obs = np.zeros_like(obs)
        # unc = env_learner.get_uncertainty(action_in=max_action * action, episode_step=episode_step)
        unc = 0
        path_uncertainty.append(unc)
        # print(new_obs[:3])
        # print(open_obs[:3])
        # print(new_obs[:3]-open_obs[:3])

        # closed_obs = open_obs
        closed.append(closed_obs)
        open2.append(open2_obs)
        open.append(open_obs)

        real.append(new_obs)
        none.append(obs)

        obs = new_obs
        episode_reward += r
        if done:
            # import torch
            # Xs = np.array([[step[0] for step in valid]], dtype=np.float32)
            # As = np.array([[step[1] for step in valid]], dtype=np.float32)
            # Ys = np.array([[step[3] for step in valid]], dtype=np.float32)
            # outs = env_learner.model(torch.from_numpy(Xs).to(device),
            #                                                                  torch.from_numpy(As).to(device),
            #                                                                  torch.from_numpy(Ys).to(device))
            # loss_seq = outs[1]
            # pred_seq = outs[-2]
            # print(loss_seq.item())
            # open_np = torch.from_numpy(np.array(open)).to(pred_seq.device)
            # pred_open_diff = open_np-pred_seq
            # print(torch.mean(torch.abs(pred_open_diff)).item())
            # Yst = torch.from_numpy(Ys).to(pred_seq.device)
            # print(torch.mean(torch.abs(pred_seq-Yst)).item())
            # print(torch.mean(torch.abs(open_np-Yst)).item())
            # open2 = pred_seq.cpu().detach().numpy()[:,0]

            real_pos = np.zeros(3)
            open_shape = [open[0].shape[i] for i in range(len(open[0].shape)-1)]
            open_shape.append(3)
            pred_pos = np.zeros(open_shape)
            pred_pos2 = np.zeros(open_shape)
            closed_pos = np.zeros(open_shape)
            real_pos_chart = []
            pred_pos_chart = []
            pred_pos2_chart = []
            closed_pos_chart = []
            decoded = []

            closed_diffs = []
            open_diffs = []
            open2_diffs = []
            print('Total Uncertainty: '+str(sum(path_uncertainty)))
            path_uncertainty = []

            for j in range(len(closed)):
                closed_diffs.append(real[j]-closed[j])
                open_diffs.append(real[j]-open[j])
                open2_diffs.append(valid[j][3]-open2[j])

                real_pos += (real[j][0:3]/0.3)/60
                real_pos_chart.append(real_pos.copy())
                if len(open[j].shape) == 1:
                    pred_pos += (open[j][0:3]/0.3)/60
                    closed_pos += (closed[j][0:3]/0.3)/60
                    pred_pos2 += (open2[j][0:3]/0.3)/60
                if len(open[j].shape) == 2:
                    pred_pos += (open[j][:,0:3]/0.3)/60
                    closed_pos += (closed[j][:,0:3]/0.3)/60
                    pred_pos2 += (open2[j][:,0:3]/0.3)/60

                pred_pos_chart.append(pred_pos.copy())
                closed_pos_chart.append(closed_pos.copy())
                pred_pos2_chart.append(pred_pos2.copy())

            # final_open_drifts.append(real[-1]-open[-1])

            losses = env_learner.get_loss(valid)
            print(losses)
            # print('Corr: '+str(losses[0])+'\tSingle: '+str(losses[1])+'\tSeq: '+str(losses[2]))
            # # print('Scaled\n\tClosed: '+str(np.mean(np.abs(np.array(closed_diffs)/np.mean(env_learner.state_mul_const))))+
            # #       '\tOpen: '+str(np.mean(np.abs(np.array(open_diffs)/np.mean(env_learner.state_mul_const)))))
            # print('Raw\n\tClosed: '+str(np.mean(np.abs(np.array(closed_diffs))))+
            #       ' \tOpen: '+str(np.mean(np.abs(np.array(open_diffs)))))
            # print('')
            # _, single, seq = env_learner.get_loss(valid)
            # print('Valid Single: ' + str(single))
            # print('Valid Seq: ' + str(seq))
            # print('Valid Corr: ' + str(corr))

            print('Closed (Green): '+str(np.mean(np.square(np.array(closed_diffs)))))
            print('Closed Stdev: '+str(np.std(closed)))
            all_closed_diffs.extend(closed_diffs)
            print('Open (Blue): '+str(np.mean(np.square(np.array(open_diffs)))))
            all_open_diffs.extend(open_diffs)
            print('Open2 (Red): '+str(np.mean(np.square(np.array(open2_diffs)))))
            all_open_diffs.extend(open2_diffs)

            all_closed_diffs.extend(closed_diffs)
            all_open_diffs.extend(open_diffs)
            if chart:
                import matplotlib.pyplot as plt
                real_pos_chart = np.array(real_pos_chart)
                xr, yr, zr = np.hsplit(real_pos_chart, 3)
                plt.plot(xr, yr, color='black')

                closed_pos_chart = np.array(closed_pos_chart)
                max_xc = 0
                max_yc = 0
                min_xc = 0
                min_yc = 0
                if len(closed_pos_chart.shape) == 2:
                    xc, yc, zc = np.hsplit(closed_pos_chart, 3)
                    plt.plot(xc, yc, color='green')
                    max_xc = max(max_xc, np.max(xc))
                    max_yc = max(max_yc, np.max(yc))
                    min_xc = min(min_xc, np.min(xc))
                    min_yc = min(min_yc, np.min(yc))
                elif len(closed_pos_chart.shape) == 3:
                    for j in range(closed_pos_chart.shape[1]):
                        xc, yc, zc = np.hsplit(closed_pos_chart[:,j,:], 3)
                        plt.plot(xc, yc, color='green')
                        max_xc = max(max_xc, np.max(xc))
                        max_yc = max(max_yc, np.max(yc))
                        min_xc = min(min_xc, np.min(xc))
                        min_yc = min(min_yc, np.min(yc))


                pred_pos_chart = np.array(pred_pos_chart)
                max_xp = 0
                max_yp = 0
                min_xp = 0
                min_yp = 0
                if len(pred_pos_chart.shape) == 2:
                    xp, yp, zp = np.hsplit(pred_pos_chart, 3)
                    plt.plot(xp, yp, color='blue')
                    max_xp = max(max_xp, np.max(xp))
                    max_yp = max(max_yp, np.max(yp))
                    min_xp = min(min_xp, np.min(xp))
                    min_yp = min(min_yp, np.min(yp))
                elif len(pred_pos_chart.shape) == 3:
                    for j in range(pred_pos_chart.shape[1]):
                        xp, yp, zp = np.hsplit(pred_pos_chart[:,j,:], 3)
                        plt.plot(xp, yp, color='blue')
                        max_xp = max(max_xp, np.max(xp))
                        max_yp = max(max_yp, np.max(yp))
                        min_xp = min(min_xp, np.min(xp))
                        min_yp = min(min_yp, np.min(yp))


                pred_pos2_chart = np.array(pred_pos2_chart)
                max_xp2 = 0
                max_yp2 = 0
                min_xp2 = 0
                min_yp2 = 0
                if len(pred_pos2_chart.shape) == 2:
                    xp2, yp2, zp2 = np.hsplit(pred_pos2_chart, 3)
                    plt.plot(xp2, yp2, color='red')
                    max_xp2 = max(max_xp2, np.max(xp2))
                    max_yp2 = max(max_yp2, np.max(yp2))
                    min_xp2 = min(min_xp2, np.min(xp2))
                    min_yp2 = min(min_yp2, np.min(yp2))
                elif len(pred_pos2_chart.shape) == 3:
                    for j in range(pred_pos2_chart.shape[1]):
                        xp2, yp2, zp2 = np.hsplit(pred_pos2_chart[:,j,:], 3)
                        plt.plot(xp2, yp2, color='red')
                        max_xp2 = max(max_xp2, np.max(xp2))
                        max_yp2 = max(max_yp2, np.max(yp2))
                        min_xp2 = min(min_xp2, np.min(xp2))
                        min_yp2 = min(min_yp2, np.min(yp2))
                # pred_pos2_chart = np.array(pred_pos2_chart)
                # xp2, yp2, zp2 = np.hsplit(pred_pos2_chart, 3)
                # plt.plot(xp2, yp2, color='red')

                max_lim = 1.1*max(np.max(xr), np.max(yr), max_xp, max_yp, max_xc, max_yc, max_xp2, max_yp2)
                min_lim = 1.1*min(np.min(xr), np.min(yr), min_xp, min_yp, min_xc, min_yc, min_xp2, min_yp2)
                plt.xlim(min_lim, max_lim)
                plt.ylim(min_lim, max_lim)
                plt.show()
                plt.clf()


            obs = env.reset()
            env_learner.reset(obs)
            open2_obs = obs

            closed = []
            open2 = []
            open = []
            real = []
            none = []
            episode_step = 0
            episode_reward = 0.
            max_ep_rew = -1000
            valid = []
            i += 1

    losses = env_learner.get_loss(all_valid)
    print('Final: ')
    print('Single: '+str(losses[0])+'\tSeq: '+str(losses[1]))
    # obs_means = np.zeros_like(all_valid[0][3])
    # mean_diffs = []
    # zero_diffs = []
    # last_diffs = []
    # for i in range(len(all_valid)):
    #     obs_means += all_valid[i][3]
    #     zero_diffs.append(all_valid[i][3])
    #     if i > 0:
    #         last_diffs.append(all_valid[i][3]-all_valid[i-1][3])
    # obs_means /= len(all_valid)
    #
    # for i in range(len(all_valid)):
    #     mean_diffs.append(obs_means - all_valid[i][3])
    # print('\nPyTorch: ')
    # print('Closed MSE: '+str(np.mean(np.square(np.array(all_closed_diffs)))))
    # print('Open MSE: '+str(np.mean(np.square(np.array(all_open_diffs)))))
    # print('Final Open: '+str(np.mean(np.square(np.array(final_open_drifts)))))
    #
    # print('Last MSE: '+str(np.mean(np.square(np.array(last_diffs)))))
    # print('Zeros MSE: '+str(np.mean(np.square(np.array(zero_diffs)))))
    # print('Variance: '+str(np.mean(np.square(np.array(mean_diffs)))))
    # print('Scaled: ')
    # print('Last MSE: '+str(np.mean(np.square(np.array(last_diffs)/env_learner.state_mul_const))))
    # print('Zeros MSE: '+str(np.mean(np.square(np.array(zero_diffs)/env_learner.state_mul_const))))
    # print('Variance: '+str(np.mean(np.square(np.array(mean_diffs)/env_learner.state_mul_const))))
    # print('Done')