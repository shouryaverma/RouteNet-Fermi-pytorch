# import tensorflow as tf


# class RouteNet_Fermi(tf.keras.Model):
#     def __init__(self):
#         super(RouteNet_Fermi, self).__init__()

#         # Configuration dictionary. It contains the needed Hyperparameters for the model.
#         # All the Hyperparameters can be found in the config.ini file

#         self.max_num_models = 7

#         self.num_policies = 4
#         self.max_num_queues = 3

#         self.iterations = 8
#         self.path_state_dim = 32
#         self.link_state_dim = 32
#         self.queue_state_dim = 32

#         self.z_score = {'traffic': [1385.4058837890625, 859.8118896484375],
#                         'packets': [1.4015231132507324, 0.8932565450668335],
#                         'eq_lambda': [1350.97119140625, 858.316162109375],
#                         'avg_pkts_lambda': [0.9117304086685181, 0.9723503589630127],
#                         'exp_max_factor': [6.663637638092041, 4.715115070343018],
#                         'pkts_lambda_on': [0.9116322994232178, 1.651275396347046],
#                         'avg_t_off': [1.6649284362792969, 2.356407403945923],
#                         'avg_t_on': [1.6649284362792969, 2.356407403945923], 'ar_a': [0.0, 1.0], 'sigma': [0.0, 1.0],
#                         'capacity': [27611.091796875, 20090.62109375], 'queue_size': [30259.10546875, 21410.095703125]}

#         # GRU Cells used in the Message Passing step
#         self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
#         self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
#         self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

#         self.path_embedding = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=10 + self.max_num_models),
#             tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu),
#             tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu)
#         ])

#         self.queue_embedding = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=self.max_num_queues + 2),
#             tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu),
#             tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu)
#         ])

#         self.link_embedding = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=self.num_policies + 1),
#             tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu),
#             tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu)
#         ])

#         self.readout_path = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(None, self.path_state_dim)),
#             tf.keras.layers.Dense(int(self.link_state_dim / 2),
#                                   activation=tf.keras.activations.relu),
#             tf.keras.layers.Dense(int(self.path_state_dim / 2),
#                                   activation=tf.keras.activations.relu),
#             tf.keras.layers.Dense(1)
#         ], name="PathReadout")

#     @tf.function
#     def call(self, inputs):
#         traffic = inputs['traffic']
#         packets = inputs['packets']
#         length = inputs['length']
#         model = inputs['model']
#         eq_lambda = inputs['eq_lambda']
#         avg_pkts_lambda = inputs['avg_pkts_lambda']
#         exp_max_factor = inputs['exp_max_factor']
#         pkts_lambda_on = inputs['pkts_lambda_on']
#         avg_t_off = inputs['avg_t_off']
#         avg_t_on = inputs['avg_t_on']
#         ar_a = inputs['ar_a']
#         sigma = inputs['sigma']

#         capacity = inputs['capacity']
#         policy = tf.one_hot(inputs['policy'], self.num_policies)

#         queue_size = inputs['queue_size']
#         priority = tf.one_hot(inputs['priority'], self.max_num_queues)
#         weight = inputs['weight']

#         queue_to_path = inputs['queue_to_path']
#         link_to_path = inputs['link_to_path']
#         path_to_link = inputs['path_to_link']
#         path_to_queue = inputs['path_to_queue']
#         queue_to_link = inputs['queue_to_link']

#         path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
#         load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

#         pkt_size = traffic / packets

#         # Initialize the initial hidden state for links
#         path_state = self.path_embedding(tf.concat(
#             [(traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
#              (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
#              tf.one_hot(model, self.max_num_models),
#              (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
#              (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
#              (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
#              (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
#              (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
#              (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
#              (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
#              (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1]], axis=1))

#         # Initialize the initial hidden state for paths
#         link_state = self.link_embedding(tf.concat([load, policy], axis=1))

#         # Initialize the initial hidden state for paths
#         queue_state = self.queue_embedding(
#             tf.concat([(queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
#                        priority, weight], axis=1))

#         # Iterate t times doing the message passing
#         for it in range(self.iterations):
#             ###################
#             #  LINK AND QUEUE #
#             #     TO PATH     #
#             ###################
#             queue_gather = tf.gather(queue_state, queue_to_path)
#             link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
#             path_update_rnn = tf.keras.layers.RNN(self.path_update,
#                                                   return_sequences=True,
#                                                   return_state=True)
#             previous_path_state = path_state

#             path_state_sequence, path_state = path_update_rnn(tf.concat([queue_gather, link_gather], axis=2),
#                                                               initial_state=path_state)

#             path_state_sequence = tf.concat([tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1)

#             ###################
#             #  PATH TO QUEUE  #
#             ###################
#             path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
#             path_sum = tf.math.reduce_sum(path_gather, axis=1)
#             queue_state, _ = self.queue_update(path_sum, [queue_state])

#             ###################
#             #  QUEUE TO LINK  #
#             ###################
#             queue_gather = tf.gather(queue_state, queue_to_link)

#             link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
#             link_state = link_gru_rnn(queue_gather, initial_state=link_state)

#         capacity_gather = tf.gather(capacity, link_to_path)
#         input_tensor = path_state_sequence[:, 1:].to_tensor()

#         ratio_gather = self.readout_path(input_tensor)
#         length = tf.ensure_shape(length, [None])
#         ratio_gather = tf.RaggedTensor.from_tensor(ratio_gather, lengths=length)

#         jitter = tf.math.reduce_sum(ratio_gather / capacity_gather,
#                                          axis=1)
#         return jitter

import torch
import torch.nn as nn


class RouteNet_Fermi(nn.Module):
    def __init__(self):
        super(RouteNet_Fermi, self).__init__()

        # Configuration parameters
        self.max_num_models = 7
        self.num_policies = 4
        self.max_num_queues = 3
        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        # Z-score normalization parameters
        self.z_score = {
            'traffic': [1385.4058837890625, 859.8118896484375],
            'packets': [1.4015231132507324, 0.8932565450668335],
            'eq_lambda': [1350.97119140625, 858.316162109375],
            'avg_pkts_lambda': [0.9117304086685181, 0.9723503589630127],
            'exp_max_factor': [6.663637638092041, 4.715115070343018],
            'pkts_lambda_on': [0.9116322994232178, 1.651275396347046],
            'avg_t_off': [1.6649284362792969, 2.356407403945923],
            'avg_t_on': [1.6649284362792969, 2.356407403945923],
            'ar_a': [0.0, 1.0],
            'sigma': [0.0, 1.0],
            'capacity': [27611.091796875, 20090.62109375],
            'queue_size': [30259.10546875, 21410.095703125]
        }

        # GRU cells for message passing
        self.path_update = nn.GRUCell(input_size=2*self.path_state_dim, hidden_size=self.path_state_dim)
        self.link_update = nn.GRUCell(input_size=self.queue_state_dim, hidden_size=self.link_state_dim)
        self.queue_update = nn.GRUCell(input_size=self.path_state_dim, hidden_size=self.queue_state_dim)

        # Neural networks for embeddings
        self.path_embedding = nn.Sequential(
            nn.Linear(10 + self.max_num_models, self.path_state_dim),
            nn.ReLU(),
            nn.Linear(self.path_state_dim, self.path_state_dim),
            nn.ReLU()
        )

        self.queue_embedding = nn.Sequential(
            nn.Linear(self.max_num_queues + 2, self.queue_state_dim),
            nn.ReLU(),
            nn.Linear(self.queue_state_dim, self.queue_state_dim),
            nn.ReLU()
        )

        self.link_embedding = nn.Sequential(
            nn.Linear(self.num_policies + 1, self.link_state_dim),
            nn.ReLU(),
            nn.Linear(self.link_state_dim, self.link_state_dim),
            nn.ReLU()
        )

        # Readout network
        self.readout_path = nn.Sequential(
            nn.Linear(self.path_state_dim, self.link_state_dim // 2),
            nn.ReLU(),
            nn.Linear(self.link_state_dim // 2, self.path_state_dim // 2),
            nn.ReLU(),
            nn.Linear(self.path_state_dim // 2, 1)
        )

    def forward(self, inputs):
        # Extract inputs
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model']
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        pkts_lambda_on = inputs['pkts_lambda_on']
        avg_t_off = inputs['avg_t_off']
        avg_t_on = inputs['avg_t_on']
        ar_a = inputs['ar_a']
        sigma = inputs['sigma']
        capacity = inputs['capacity']
        policy = torch.nn.functional.one_hot(inputs['policy'], self.num_policies)
        queue_size = inputs['queue_size']
        priority = torch.nn.functional.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']

        # Connectivity matrices
        queue_to_path = inputs['queue_to_path']
        link_to_path = inputs['link_to_path']
        path_to_link = inputs['path_to_link']
        path_to_queue = inputs['path_to_queue']
        queue_to_link = inputs['queue_to_link']

        # Calculate load
        path_gather_traffic = torch.gather(traffic, 1, path_to_link[:, :, 0])
        load = torch.sum(path_gather_traffic, dim=1) / capacity

        # Initialize states
        path_state = self.path_embedding(torch.cat([
            (traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
            (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
            torch.nn.functional.one_hot(model, self.max_num_models),
            (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
            (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
            (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
            (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
            (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
            (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
            (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
            (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1]
        ], dim=1))

        link_state = self.link_embedding(torch.cat([load, policy], dim=1))
        
        queue_state = self.queue_embedding(torch.cat([
            (queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
            priority,
            weight
        ], dim=1))

        # Message passing iterations
        path_states_sequence = []
        for _ in range(self.iterations):
            previous_path_state = path_state.clone()
            
            # Link and Queue to Path
            queue_gather = torch.gather(queue_state, 1, queue_to_path.expand(-1, -1, self.queue_state_dim))
            link_gather = torch.gather(link_state, 1, link_to_path.expand(-1, -1, self.link_state_dim))
            
            # Update path state
            combined_message = torch.cat([queue_gather, link_gather], dim=2)
            path_state = self.path_update(
                combined_message.view(-1, 2*self.path_state_dim),
                path_state.view(-1, self.path_state_dim)
            ).view(path_state.shape)
            
            path_states_sequence.append(path_state)

            # Path to Queue
            path_gather = torch.gather(path_state, 1, path_to_queue.expand(-1, -1, self.path_state_dim))
            path_sum = torch.sum(path_gather, dim=1)
            queue_state = self.queue_update(path_sum, queue_state)

            # Queue to Link
            queue_gather = torch.gather(queue_state, 1, queue_to_link.expand(-1, -1, self.queue_state_dim))
            link_state = self.link_update(
                queue_gather.view(-1, self.queue_state_dim),
                link_state.view(-1, self.link_state_dim)
            ).view(link_state.shape)

        # Readout
        path_states_tensor = torch.stack(path_states_sequence, dim=1)
        capacity_gather = torch.gather(capacity, 1, link_to_path)
        
        ratio_gather = self.readout_path(path_states_tensor)
        # Handle variable length sequences
        mask = torch.arange(ratio_gather.size(1)).expand(length.size(0), -1) < length.unsqueeze(1)
        ratio_gather = ratio_gather * mask.unsqueeze(-1).float()
        
        jitter = torch.sum(ratio_gather / capacity_gather.unsqueeze(-1), dim=1)
        return jitter