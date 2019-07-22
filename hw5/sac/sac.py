import tensorflow as tf
import time
import utils
import numpy as np


DEBUG = False


class SAC:
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 pool,
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01,
                 use_model=False,
                 horizon=15,
                 num_action_selection=5,
                 **kwargs):
        """
        Args:
        """
        self._pool = pool

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau
        self._use_model = use_model
        self._horizon = horizon
        self._num_action_selection = num_action_selection
        self._best_actions = None
        
        self._training_loss = {}
        self._loss_show = {}
        self._training_ops = []
        self._target_update_ops = None
        self._debug_val = {}
        self._debug_val_show = {}

    def add_debug(self, name, x):
        if DEBUG:
            self._debug_val[name] = tf.identity(x, name=name+'_debug')
    
    def _add_loss(self, name, x):
        self._training_loss[name] = tf.identity(x, name=name)
        
    def build(self, env, policy, q_function, q_function2, value_function,
              target_value_function, model_function):
        
        self._create_placeholders(env)
        
        value_function_loss = self._value_function_loss_for(
            policy, q_function, q_function2, value_function)
        q_function_loss = self._q_function_loss_for(q_function,
                                                    target_value_function)
        if q_function2 is not None:
            q_function2_loss = self._q_function_loss_for(q_function2,
                                                         target_value_function)
        if self._use_model:
            obs_pre_loss, rewards_pre_loss = self._model_loss_for(model_function)
            
        policy_loss = self._policy_loss_for(policy, q_function, q_function2, value_function, model_function)
        
        optimizer = tf.train.AdamOptimizer(
            self._learning_rate, name='optimizer')
        policy_training_op = optimizer.minimize(
            loss=policy_loss, var_list=policy.trainable_variables)
        value_training_op = optimizer.minimize(
            loss=value_function_loss,
            var_list=value_function.trainable_variables)
        q_function_training_op = optimizer.minimize(
            loss=q_function_loss, var_list=q_function.trainable_variables)
        

        self._training_ops = [
            policy_training_op, value_training_op, q_function_training_op
        ]
        
        self._add_loss('policy_loss', policy_loss)
        self._add_loss('value_function_loss', value_function_loss)
        self._add_loss('q_function_loss', q_function_loss)


        if self._use_model:
            _, _, self._best_actions = self._setup_action_selection(model_function, policy, target_value_function)
            
            obs_prd_training_op = optimizer.minimize(
                loss=obs_pre_loss, var_list=model_function.trainable_variables)
            rewards_ped_training_op = optimizer.minimize(
                loss=rewards_pre_loss, var_list=model_function.trainable_variables)
            
            self._training_ops += [obs_prd_training_op, rewards_ped_training_op]
            self._add_loss('obs_pre_loss', obs_pre_loss)
            self._add_loss('rewards_pre_loss', rewards_pre_loss)
            
        if q_function2 is not None:
            q_function2_training_op = optimizer.minimize(
                loss=q_function2_loss, var_list=q_function2.trainable_variables)
            self._training_ops += [q_function2_training_op]
            self._add_loss('q_function2_loss', q_function2_loss)
        
        self._target_update_ops = self._create_target_update(
            source=value_function, target=target_value_function)

        tf.get_default_session().run(tf.global_variables_initializer())

    def _create_placeholders(self, env):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, action_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    def _model_predict(self, model_function, observations, actions):
        observations_norm = utils.normalize(observations, self._pool.observations_mean, self._pool.observations_std)
        actions_norm = utils.normalize(actions, self._pool.actions_mean, self._pool.actions_std)
        
        delta_norm, rewards = model_function([observations_norm, actions_norm])
        
        delta = utils.unnormalize(delta_norm, self._pool.delta_observations_mean, self._pool.delta_observations_std)
        next_observations = observations + delta
        
        return next_observations, tf.squeeze(rewards)

    def _model_loss_for(self, model_function):
        next_obs_pred, reward_pre = self._model_predict(model_function, self._observations_ph, self._actions_ph)
        obs_diff = self._next_observations_ph - self._observations_ph
        obs_pred_diff = next_obs_pred - self._observations_ph
        obs_diff_norm = utils.normalize(obs_diff,
                                        self._pool.delta_observations_mean,
                                        self._pool.delta_observations_std)
        obs_pred_diff_norm = utils.normalize(obs_pred_diff,
                                             self._pool.delta_observations_mean,
                                             self._pool.delta_observations_std)
        
        obs_pre_loss = tf.losses.mean_squared_error(obs_diff_norm, obs_pred_diff_norm)
        reward_pre_loss = tf.losses.mean_squared_error(self._rewards_ph, reward_pre)
        
        return obs_pre_loss, reward_pre_loss

    def _setup_action_selection(self, model_function, policy, value_function):
        batch_size = 1
        if model_function is None:
            return tf.random_uniform([batch_size, 1], -1, 1)
        else:
            observations = tf.tile(self._observations_ph, [self._num_action_selection, 1])
        
            # shape: [batch_size * num_action_selection, 1]
            rewards = tf.zeros(self._num_action_selection * batch_size, dtype=tf.float32)
        
            actions, log_pis = policy(observations)
            action_list = [tf.gather(actions, tf.range(self._num_action_selection) * batch_size + i, axis=0)
                           for i in range(self._batch_size)]
            temp_log_pis = tf.identity(log_pis, name='temp_log_pis')
        
            for i in range(self._horizon):
                observations, temp_rewards = self._model_predict(model_function, observations, actions)
                rewards += tf.pow(self._discount, i) * (tf.squeeze(temp_rewards) - self._alpha * temp_log_pis)
                actions, temp_log_pis = policy(observations)
        
            rewards += tf.pow(self._discount, self._horizon) * tf.squeeze(value_function(observations))
            # rewards = tf.squeeze(rewards)
        
            # list of tensor [num_action_selection]. length = batch size
            rewards_list = [tf.gather(rewards, tf.range(self._num_action_selection) * batch_size + i)
                            for i in range(batch_size)]
        
            # list of tensor [num_action_selection]. length = batch size
            log_pis_list = [tf.gather(log_pis, tf.range(self._num_action_selection) * batch_size + i, axis=0)
                            for i in range(batch_size)]
        
            # list of tf.float32 with length = batch size
            best_action_indices = [tf.argmax(r) for r in rewards_list]
        
            # list of tf.float32 with length = batch size
            best_action_log_pis = [tf.gather(log_pis_list[i], best_action_indices[i], axis=0)
                                   for i in range(batch_size)]

            max_reward_list = [tf.gather(rewards_list[i], best_action_indices[i], axis=0)
                               for i in range(batch_size)]
            best_action_list = [tf.gather(action_list[i], best_action_indices[i], axis=0)
                                for i in range(batch_size)]
            
            self.add_debug('best_action', tf.stack(best_action_list, axis=0))

            # shape = [batch size, 1], action shape is [batch size, action_dim]
            return tf.stack(best_action_log_pis, axis=0),\
                   tf.stack(max_reward_list, axis=0), \
                   tf.stack(best_action_list, axis=0)

    def _policy_loss_for(self, policy, q_function, q_function2, value_function, model_function):
        if q_function2 is None:
            q_function2 = q_function
        
        if not self._reparameterize:
            ### Problem 1.3.A
            ### YOUR CODE HERE
            action, log_pis = policy(self._observations_ph)
            loss = log_pis * tf.stop_gradient(self._alpha * log_pis
                                              - tf.squeeze(tf.minimum(q_function([self._observations_ph, action]),
                                                                      q_function2([self._observations_ph, action])))
                                              + tf.squeeze(value_function(self._observations_ph)))
        else:
            ### Problem 1.3.B
            ### YOUR CODE HERE
            action, log_pis = policy(self._observations_ph)
            self.add_debug('log_pis', log_pis)
            loss = self._alpha * log_pis - tf.squeeze(tf.minimum(q_function([self._observations_ph, action]),
                                                                 q_function2([self._observations_ph, action])))
        
        return tf.reduce_mean(loss)

    def _value_function_loss_for(self, policy, q_function, q_function2, value_function):
        ### Problem 1.2.A
        ### YOUR CODE HERE
        if q_function2 is None:
            q_function2 = q_function

        action, log_pis = policy(self._observations_ph)
        target_value = (tf.squeeze(tf.minimum(q_function([self._observations_ph, action]),
                                              q_function2([self._observations_ph, action])))
                        - self._alpha * log_pis)
        loss = (tf.squeeze(value_function(self._observations_ph)) - target_value) ** 2
        
        # self.add_debug('target_value', target_value)
        # self.add_debug('value', tf.squeeze(value_function(self._observations_ph)))
        # self.add_debug('value_loss', tf.reduce_mean(loss))

        return tf.reduce_mean(loss)

    def _q_function_loss_for(self, q_function, target_value_function):
        ### Problem 1.1.A
        ### YOUR CODE HERE
        loss = (tf.squeeze(q_function([self._observations_ph, self._actions_ph]))
                - (self._rewards_ph + (1 - self._terminals_ph)
                   * self._discount
                   * tf.squeeze(target_value_function(self._next_observations_ph)))
                ) ** 2
        
        return tf.reduce_mean(loss)

    def _create_target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        return [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target.trainable_variables, source.
                                      trainable_variables)
        ]
    
    def get_actions_using_model(self, observations):
        if np.ndim(observations) == 1:
            observations = [observations]
            
        feed_dict = {self._observations_ph: observations}
        best_actions = tf.get_default_session().run(self._best_actions, feed_dict)
        return best_actions

    def train(self, sampler, n_epochs=1000):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample(policy=self.get_actions_using_model)
    
                batch = sampler.random_batch(self._batch_size)
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }
                self._debug_val_show, self._loss_show, _ = tf.get_default_session().run(
                    [self._debug_val, self._training_loss, self._training_ops], feed_dict)
                tf.get_default_session().run(self._target_update_ops)
            
                # For debug
                if DEBUG:
                    print('epoch: {}, t: {}\n'.format(epoch, t))
                    for name, val in self._debug_val_show.items():
                        print('{} is {}\n'.format(name, val))
                    print('-'*50 + '\n')
            
            yield epoch

    def get_statistics(self):
        statistics = {
            'Time': time.time() - self._start,
            'TimestepsThisBatch': self._epoch_length,
        }
        
        for name, val in self._loss_show.items():
            statistics[name] = val
            

        return statistics
