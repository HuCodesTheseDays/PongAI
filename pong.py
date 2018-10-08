# Takes 3 nights to train

import numpy as np
import cPickle as pickle
import gym

# hyper parameters
H = 200  # number of hidden layers
batch_size = 10  # every how many episodes to do a parameter update?
learning_rate = 1e-4  # for convergence (too slow- slow to converge, too high- never converge)
gamma = 0.99  # discount factor for reward (i.e later rewards are exponentially less important)
decay_rate = 0.99  # decay factor for RMSProp leakky sum of grad^2
resume = False  # resume from previous checkpoint?

# initialize model
D = 80*80  # input dimensionality: 80x80 grid (pong world)
if resume:
    model = pickle.load(open('save.p', 'rb'))  # load from pickled checkpoint
else:
    model = {}  # initalize model
    # xavier initialization
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)
grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}


def sigmoid(x):  # activation function
    return 1.0/(1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


# takes a single game frame as input
# preprocesses before feeding into model
def prepro(input):
    input = input[35:195]  # crop the screen
    input = input[::2, ::2, 0]  # downsample by factor of 2
    input[input == 144] = 0  # erase background (background type 1)
    input[input == 109] = 0  # erase background (background type 2)
    input[input != 0] = 1  # paddles and ball set to 1
    return input.astype(np.float).ravel()  # flatten into 1d array


def discount_rewards(r):
    # take 1d float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)  # initialize discount reward matrix as empty
    running_add = 0  # store reward sums
    for t in reversed(xrange(0, r.size)):
        # if reward at index t is nonzero, reset the sum, since this was a game boundary (pong specific)
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):  # forward propogation via numpy
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h


def policy_backward(eph, epdlogp):
    # eph is array of interemeditate states
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    # return both derivatives to update weights
    return {'W1': dW1, 'W2': dW2}


# environment
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0
    dlogps.append(y-aprob)

    env.render()
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_number += 1

        epx = np.vstack(xs)  # observation
        eph = np.vstack(hs)  # hidden
        epdlogp = np.vstack(dlogps)  # gradient
        epr = np.vstack(drs)  # reward
        xs, hs, dlogps, drs = [], [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1-decay_rate) * g**2
                model[k] += learning_rate*g/(np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('setting env. episode reward total was %f. running mean: %f' %
              (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()
        prev_x = None

    if reward != 0:
        print ('ep %d: game finished, reward: %f' %
               (episode_number, reward)) + ('' if reward == -1 else ' !!!')
