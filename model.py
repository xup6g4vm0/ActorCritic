import numpy as np
import tensorflow as tf

class ActorCritic:
  def __init__(
          self,
          state_dims,
          action_dims,
          learning_rate_actor=0.001,
          learning_rate_critic=0.01,
          reward_decay=0.9,
          output_graph=False
  ):
    self.state_dims = state_dims
    self.action_dims = action_dims
    self.lra = learning_rate_actor
    self.lrc = learning_rate_critic
    self.gamma = reward_decay
    
    self._build_net()

    self.sess = tf.Session()

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

  def _build_net(self):
    with tf.variable_scope('actor'):
      self.s = tf.placeholder(tf.float32, [1, self.state_dims], name='state')
      self.a = tf.placeholder(tf.int32, None, name='actions')
      self.td_error = tf.placeholder(tf.float32, None, name='td_error')

      with tf.variable_scope('network'):
        h1 = tf.layers.dense(self.s, 20, activation=tf.nn.relu)
        all_act = tf.layers.dense(h1, self.action_dims, activation=None)

      self.all_act_prob = tf.nn.softmax(all_act)

      with tf.variable_scope('loss'):
        log_prob = tf.log(self.all_act_prob[0, self.a])
        self.aloss = -tf.reduce_mean(log_prob * self.td_error)
      with tf.variable_scope('train'):
        self.atrain_op = tf.train.AdamOptimizer(self.lra).minimize(self.aloss)

    with tf.variable_scope('critic'):
      self.s_ = tf.placeholder(tf.float32, [1, self.state_dims], name='nstate')
      self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_next')
      self.r = tf.placeholder(tf.float32, None, name='reward')

      with tf.variable_scope('network'):
        h1 = tf.layers.dense(self.s_, 20, activation=tf.nn.relu)
        self.v = tf.layers.dense(h1, 1, activation=None)

      with tf.variable_scope('squared_TD_error'):
        self.ctd_error = self.r + self.gamma * self.v_ - self.v
        self.closs = tf.square(self.ctd_error)
      with tf.variable_scope('train'):
        self.ctrain_op = tf.train.AdamOptimizer(self.lrc).minimize(self.closs)

  def choose_action(self, state, _eval=False):
    state = state[np.newaxis, :]

    action_prob = self.sess.run(self.all_act_prob, feed_dict={
                      self.s: state })
    action = np.random.choice(self.action_dims, p=action_prob[0])

    return action

  def learn(self, s, a, r, ns):
    s, ns = s[np.newaxis, :], ns[np.newaxis, :]
    
    # update critic
    nv = self.sess.run(self.v, feed_dict={ self.s_: ns })
    _, tde = self.sess.run([self.ctrain_op, self.ctd_error], feed_dict={
                      self.s_: s,
                      self.v_: nv,
                      self.r: r })

    # update actor
    self.sess.run(self.atrain_op, feed_dict={
        self.s: s,
        self.a: a,
        self.td_error: tde})

