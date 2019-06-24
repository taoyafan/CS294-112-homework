import argparse
import gym
import load_policy
import os
import tensorflow as tf
import numpy as np

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--exp_name', type=str, default='temp')
    parser.add_argument('--log_root', type=str)
    parser.add_argument("--hidden_dim", type=int, default=20)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=200,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    return args


class Model:
    def __init__(self, args, env):
        self.args = args
        self.sate_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        train_dir = os.path.join(args.log_root, args.exp_name, args.env, str(args.hidden_dim), "train")
        print("Train dir is ", train_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
        self.sv = tf.train.Supervisor(logdir=train_dir,
                           is_chief=True,
                           saver=self.saver,
                           summary_op=None,
                           save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                           save_model_secs=60,      # checkpoint every 60 secs
                           global_step=self.global_step,
                           init_feed_dict=None
                           )
        self.summary_writer = self.sv.summary_writer
        self.sess = self.sv.prepare_or_wait_for_session()
        self.global_step = self.sess.run(tf.train.get_global_step())
        
    def build_model(self):
        print('Building graph')
        self._obs_in = tf.placeholder(tf.float32, [None, self.sate_dim], name='obs_in')
        self._target_action = tf.placeholder(tf.float32, [None, self.action_dim], name='target_action')
        
        # build network
        self.fc1 = tf.contrib.layers.fully_connected(self._obs_in, self.args.hidden_dim,
                                                     activation_fn=tf.nn.leaky_relu,
                                                     weights_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                                     biases_initializer=tf.zeros_initializer)

        self._output = tf.contrib.layers.fully_connected(self.fc1,
                                                              self.action_dim,
                                                              activation_fn=None,
                                                              weights_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                                              biases_initializer=tf.zeros_initializer)
        # Calculate the loss
        self.cost = tf.reduce_mean(tf.square(self._target_action - self._output))
        tf.summary.scalar('cost', self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(0.01, 0.99, 0.0, 1e-6).minimize(self.cost, global_step=self.global_step)
        self.summaries = tf.summary.merge_all()
        
    def train(self, obs, target):
        if obs.ndim == 1:
            obs = np.expand_dims(obs, 0)
            
        cost, _, self.global_step, summaries = self.sess.run(
            [self.cost, self.optimizer, tf.train.get_global_step(), self.summaries],
             feed_dict={self._obs_in: obs, self._target_action: target})
        
        self.summary_writer.add_summary(summaries, self.global_step)
        return cost
        
    def predict(self, obs):
        obs = np.array(obs)

        # Ensure state.ndim == 2
        if obs.ndim == 1:
            obs = np.expand_dims(obs, 0)
            
        return self.sess.run(self._output, feed_dict={self._obs_in: obs})
    
    def record_reward(self, r):
        reward = tf.Summary()
        reward.value.add(tag='reward/train_step', simple_value=r)
        self.summary_writer.add_summary(reward, self.global_step)
        return


def main():
    args = init_config()
    env = gym.make(args.env)

    print('state_dim = ', env.observation_space.shape)
    print('action_dim = ', env.action_space.shape)

    max_steps = args.max_timesteps or env.spec.timestep_limit
    policy_fn = load_policy.load_policy('experts/'+args.env+'.pkl')
    agent = Model(args, env)

    with agent.sess:
        for i in range(args.num_rollouts):
            print('Iter:', i)
            obs = env.reset()
            steps = 0
            done = False
            loss = 0
            r_sum = 0
    
            while not done:
                if i % 10 == 0:
                    env.render()
                
                action = policy_fn(obs[None, :])
                loss += agent.train(obs, action)
                obs, r, done, _ = env.step(agent.predict(obs))
                r_sum += r
                steps += 1
    
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
                    
            agent.record_reward(r_sum)
            print("Average loss is {}".format(loss/steps))
            print('Tested policy get {} reward'.format(r_sum))
            


if __name__ == '__main__':
    main()
