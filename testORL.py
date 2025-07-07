import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict, deque
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class SyntheticOpioidDataGenerator:
    def __init__(self, n_patients=1000, max_episode_length=52):
        self.n_patients = n_patients
        self.max_episode_length = max_episode_length  

    def generate_patient_features(self) -> Dict:
        patients = []

        for i in range(self.n_patients):

            age = np.random.normal(55, 15)
            age = np.clip(age, 18, 85)

            initial_dose = np.random.lognormal(3.5, 0.8)  
            initial_dose = np.clip(initial_dose, 10, 300)

            duration_of_use = np.random.exponential(24)  
            duration_of_use = np.clip(duration_of_use, 1, 120)

            depression = np.random.binomial(1, 0.3)
            anxiety = np.random.binomial(1, 0.25)
            substance_abuse_history = np.random.binomial(1, 0.15)
            chronic_pain = np.random.binomial(1, 0.8)

            baseline_pain = np.random.normal(7, 1.5)
            baseline_pain = np.clip(baseline_pain, 1, 10)

            withdrawal_risk = (
                0.3 * (initial_dose / 100) + 
                0.2 * (duration_of_use / 60) + 
                0.2 * depression + 
                0.15 * anxiety + 
                0.15 * substance_abuse_history
            )

            patient = {
                'patient_id': i,
                'age': age,
                'initial_dose': initial_dose,
                'duration_of_use': duration_of_use,
                'depression': depression,
                'anxiety': anxiety,
                'substance_abuse_history': substance_abuse_history,
                'chronic_pain': chronic_pain,
                'baseline_pain': baseline_pain,
                'withdrawal_risk': withdrawal_risk
            }
            patients.append(patient)

        return pd.DataFrame(patients)

    def simulate_episode(self, patient_row: pd.Series) -> List[Dict]:
        episode = []

        current_dose = patient_row['initial_dose']
        current_pain = patient_row['baseline_pain']
        withdrawal_severity = 0
        week = 0

        base_aggressiveness = np.random.normal(0.12, 0.03)  
        base_aggressiveness = np.clip(base_aggressiveness, 0.05, 0.25)

        physician_confidence = 0.5  

        while week < self.max_episode_length and current_dose > 0:

            state = {
                'week': week,
                'current_dose': current_dose,
                'current_pain': current_pain,
                'withdrawal_severity': withdrawal_severity,
                'age': patient_row['age'],
                'depression': patient_row['depression'],
                'anxiety': patient_row['anxiety'],
                'substance_abuse_history': patient_row['substance_abuse_history'],
                'chronic_pain': patient_row['chronic_pain'],
                'baseline_pain': patient_row['baseline_pain'],
                'withdrawal_risk': patient_row['withdrawal_risk'],
                'dose_change_last_week': 0 if week == 0 else episode[-1]['dose_change']
            }

            if week == 0:
                dose_change = 0  
            else:

                base_reduction = base_aggressiveness * current_dose

                pain_factor = 1.0
                if current_pain > patient_row['baseline_pain'] + 2:
                    pain_factor = 0.3  
                elif current_pain > patient_row['baseline_pain'] + 1:
                    pain_factor = 0.6  

                withdrawal_factor = 1.0
                if withdrawal_severity > 7:
                    withdrawal_factor = 0.2  
                elif withdrawal_severity > 4:
                    withdrawal_factor = 0.5  

                if week > 4:
                    recent_pain_trend = current_pain - episode[-4]['current_pain']
                    if recent_pain_trend > 1:
                        physician_confidence *= 0.8  
                    elif recent_pain_trend < -0.5:
                        physician_confidence *= 1.1  

                physician_confidence = np.clip(physician_confidence, 0.2, 1.0)

                dose_change = -base_reduction * pain_factor * withdrawal_factor * physician_confidence

                noise = np.random.normal(0, abs(dose_change) * 0.15)
                dose_change += noise

                dose_change = max(dose_change, -current_dose)
                dose_change = min(dose_change, current_dose * 0.3)  

            new_dose = current_dose + dose_change
            new_dose = max(0, new_dose)

            dose_reduction_ratio = abs(dose_change) / max(current_dose, 1)

            if dose_reduction_ratio > 0:

                hyperalgesia_effect = dose_reduction_ratio * np.random.normal(1.5, 0.3)
                hyperalgesia_effect = max(0, hyperalgesia_effect)

                tolerance_effect = (patient_row['initial_dose'] / 100) * dose_reduction_ratio * np.random.normal(0.5, 0.2)
                tolerance_effect = max(0, tolerance_effect)

                pain_increase = hyperalgesia_effect + tolerance_effect
            else:
                pain_increase = 0

            if dose_reduction_ratio > 0:
                withdrawal_base = dose_reduction_ratio * np.random.normal(2.5, 0.5)
                withdrawal_multiplier = patient_row['withdrawal_risk'] * 2
                withdrawal_increase = withdrawal_base * withdrawal_multiplier
                withdrawal_increase = max(0, withdrawal_increase)
            else:
                withdrawal_increase = 0

            pain_recovery = np.random.normal(0.08, 0.03)  
            withdrawal_recovery = np.random.normal(0.15, 0.05)  

            current_pain += pain_increase - pain_recovery
            current_pain = np.clip(current_pain, 0, 10)

            withdrawal_severity += withdrawal_increase - withdrawal_recovery
            withdrawal_severity = np.clip(withdrawal_severity, 0, 10)

            dose_reduction_reward = 0
            if dose_change < 0:
                dose_reduction_reward = abs(dose_change) / patient_row['initial_dose'] * 20

            pain_above_baseline = max(0, current_pain - patient_row['baseline_pain'])
            pain_penalty = -(pain_above_baseline ** 1.5) * 3

            withdrawal_penalty = -(withdrawal_severity ** 1.2) * 2

            completion_bonus = 0
            if new_dose == 0:
                completion_bonus = 100
            elif new_dose < patient_row['initial_dose'] * 0.1:  
                completion_bonus = 30

            stability_bonus = 0
            if current_pain > 8 or withdrawal_severity > 6:
                if abs(dose_change) < current_dose * 0.05:  
                    stability_bonus = 10

            reward = dose_reduction_reward + pain_penalty + withdrawal_penalty + completion_bonus + stability_bonus

            dropout_prob = 0.005  
            dropout_prob += 0.02 * (current_pain / 10) ** 2  
            dropout_prob += 0.025 * (withdrawal_severity / 10) ** 2  

            if dose_reduction_ratio > 0.2:
                dropout_prob += 0.03

            terminated = np.random.random() < dropout_prob

            episode_step = {
                **state,
                'dose_change': dose_change,
                'new_dose': new_dose,
                'new_pain': current_pain,
                'new_withdrawal': withdrawal_severity,
                'reward': reward,
                'terminated': terminated,
                'patient_id': patient_row['patient_id']
            }

            episode.append(episode_step)

            current_dose = new_dose
            week += 1

            if terminated or current_dose == 0:
                break

        return episode

        physician_aggressiveness = np.random.normal(0.15, 0.05)  
        physician_aggressiveness = np.clip(physician_aggressiveness, 0.05, 0.3)

        while week < self.max_episode_length and current_dose > 0:

            state = {
                'week': week,
                'current_dose': current_dose,
                'current_pain': current_pain,
                'withdrawal_severity': withdrawal_severity,
                'age': patient_row['age'],
                'depression': patient_row['depression'],
                'anxiety': patient_row['anxiety'],
                'substance_abuse_history': patient_row['substance_abuse_history'],
                'chronic_pain': patient_row['chronic_pain'],
                'baseline_pain': patient_row['baseline_pain'],
                'withdrawal_risk': patient_row['withdrawal_risk'],
                'dose_change_last_week': 0 if week == 0 else episode[-1]['dose_change']
            }

            if week == 0:
                dose_change = 0  
            else:

                base_reduction = physician_aggressiveness * current_dose

                if current_pain > 8:
                    reduction_factor = 0.5  
                elif withdrawal_severity > 6:
                    reduction_factor = 0.3  
                else:
                    reduction_factor = 1.0

                dose_change = -base_reduction * reduction_factor

                dose_change += np.random.normal(0, base_reduction * 0.1)

                dose_change = max(dose_change, -current_dose)

            new_dose = current_dose + dose_change
            new_dose = max(0, new_dose)

            dose_reduction_ratio = abs(dose_change) / max(current_dose, 1)
            pain_increase = dose_reduction_ratio * np.random.normal(2, 0.5)
            pain_increase = max(0, pain_increase)

            withdrawal_increase = dose_reduction_ratio * np.random.normal(3, 1) * patient_row['withdrawal_risk']
            withdrawal_increase = max(0, withdrawal_increase)

            pain_recovery = np.random.normal(0.1, 0.05)
            withdrawal_recovery = np.random.normal(0.2, 0.1)

            current_pain += pain_increase - pain_recovery
            current_pain = np.clip(current_pain, 0, 10)

            withdrawal_severity += withdrawal_increase - withdrawal_recovery
            withdrawal_severity = np.clip(withdrawal_severity, 0, 10)

            dose_reduction_reward = abs(dose_change) / patient_row['initial_dose'] * 10
            pain_penalty = -(current_pain - patient_row['baseline_pain']) * 2
            withdrawal_penalty = -withdrawal_severity * 1.5

            completion_bonus = 50 if new_dose == 0 else 0

            reward = dose_reduction_reward + pain_penalty + withdrawal_penalty + completion_bonus

            dropout_prob = 0.01 + 0.02 * (current_pain / 10) + 0.03 * (withdrawal_severity / 10)
            terminated = np.random.random() < dropout_prob

            episode_step = {
                **state,
                'dose_change': dose_change,
                'new_dose': new_dose,
                'new_pain': current_pain,
                'new_withdrawal': withdrawal_severity,
                'reward': reward,
                'terminated': terminated,
                'patient_id': patient_row['patient_id']
            }

            episode.append(episode_step)

            current_dose = new_dose
            week += 1

            if terminated or current_dose == 0:
                break

        return episode

    def generate_dataset(self) -> pd.DataFrame:
        patients = self.generate_patient_features()
        all_episodes = []

        print(f"Generating episodes for {self.n_patients} patients...")

        for idx, patient in patients.iterrows():
            episode = self.simulate_episode(patient)
            all_episodes.extend(episode)

            if (idx + 1) % 100 == 0:
                print(f"Generated {idx + 1} patient episodes")

        dataset = pd.DataFrame(all_episodes)
        print(f"Generated {len(dataset)} total transitions")

        return dataset, patients

class MDPEnvironment:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.state_features = [
            'week', 'current_dose', 'current_pain', 'withdrawal_severity',
            'age', 'depression', 'anxiety', 'substance_abuse_history',
            'chronic_pain', 'baseline_pain', 'withdrawal_risk', 'dose_change_last_week'
        ]

        self.state_scaler = StandardScaler()
        self.state_scaler.fit(dataset[self.state_features])

        self.action_bins = np.linspace(-50, 10, 21)  

    def get_state(self, row: pd.Series) -> np.ndarray:
        state = row[self.state_features].values.reshape(1, -1)
        return self.state_scaler.transform(state)[0]

    def discretize_action(self, dose_change: float) -> int:
        return np.digitize(dose_change, self.action_bins) - 1

    def continuous_action(self, action_idx: int) -> float:
        return self.action_bins[min(action_idx, len(self.action_bins) - 1)]

class ReplayBuffer:
    def __init__(self, dataset: pd.DataFrame, env: MDPEnvironment):
        self.transitions = []
        self.env = env

        for _, row in dataset.iterrows():
            state = env.get_state(row)
            action = env.discretize_action(row['dose_change'])
            reward = row['reward']

            next_row = dataset[
                (dataset['patient_id'] == row['patient_id']) & 
                (dataset['week'] == row['week'] + 1)
            ]

            if len(next_row) > 0:
                next_state = env.get_state(next_row.iloc[0])
                done = next_row.iloc[0]['terminated']
            else:
                next_state = state  
                done = True

            self.transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

    def sample(self, batch_size: int) -> Dict:
        indices = np.random.choice(len(self.transitions), batch_size, replace=False)
        batch = [self.transitions[i] for i in indices]

        return {
            'states': torch.FloatTensor([t['state'] for t in batch]),
            'actions': torch.LongTensor([t['action'] for t in batch]),
            'rewards': torch.FloatTensor([t['reward'] for t in batch]),
            'next_states': torch.FloatTensor([t['next_state'] for t in batch]),
            'dones': torch.tensor([bool(t['done']) for t in batch], dtype=torch.bool)
        }

    def __len__(self):
        return len(self.transitions)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.network(state)

class CQLAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, 
                 alpha: float = 0.5, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  
        self.gamma = gamma

        self.q_net1 = QNetwork(state_dim, action_dim)
        self.q_net2 = QNetwork(state_dim, action_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)

        self.max_grad_norm = 1.0

    def update_target_network(self, tau: float = 0.005):
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_step(self, batch: Dict) -> Dict:
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        current_q1 = self.q_net1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.q_net2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            next_q1 = self.target_q_net1(next_states)
            next_q2 = self.target_q_net2(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_q_values = next_q.max(1)[0]

            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            target_q_values = torch.clamp(target_q_values, -100, 100)

        bellman_error1 = nn.SmoothL1Loss()(current_q1, target_q_values)
        bellman_error2 = nn.SmoothL1Loss()(current_q2, target_q_values)

        q_values1_all = self.q_net1(states)
        q_values2_all = self.q_net2(states)

        cql_loss1 = torch.logsumexp(q_values1_all, dim=1).mean() - current_q1.mean()
        cql_loss2 = torch.logsumexp(q_values2_all, dim=1).mean() - current_q2.mean()

        total_loss1 = bellman_error1 + self.alpha * cql_loss1
        total_loss2 = bellman_error2 + self.alpha * cql_loss2

        self.optimizer1.zero_grad()
        total_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.max_grad_norm)
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        total_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.max_grad_norm)
        self.optimizer2.step()

        return {
            'total_loss1': total_loss1.item(),
            'total_loss2': total_loss2.item(),
            'bellman_error1': bellman_error1.item(),
            'bellman_error2': bellman_error2.item(),
            'cql_loss1': cql_loss1.item(),
            'cql_loss2': cql_loss2.item()
        }

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values1 = self.q_net1(state_tensor)
            q_values2 = self.q_net2(state_tensor)
            q_values = torch.min(q_values1, q_values2)
            return q_values.argmax().item()

class BCQAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, 
                 gamma: float = 0.99, threshold: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.threshold = threshold

        self.q_net1 = QNetwork(state_dim, action_dim)
        self.q_net2 = QNetwork(state_dim, action_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.bc_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.bc_optimizer = optim.Adam(self.bc_net.parameters(), lr=lr)

        self.max_grad_norm = 1.0

    def update_target_network(self, tau: float = 0.005):
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_step(self, batch: Dict) -> Dict:
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        bc_logits = self.bc_net(states)
        bc_loss = nn.CrossEntropyLoss()(bc_logits, actions)

        self.bc_optimizer.zero_grad()
        bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.bc_net.parameters(), self.max_grad_norm)
        self.bc_optimizer.step()

        current_q1 = self.q_net1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = self.q_net2(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            next_bc_logits = self.bc_net(next_states)
            next_bc_probs = torch.softmax(next_bc_logits, dim=1)

            next_actions_mask = next_bc_probs > self.threshold

            next_q1 = self.target_q_net1(next_states)
            next_q2 = self.target_q_net2(next_states)
            next_q = torch.min(next_q1, next_q2)

            next_q_masked = next_q.masked_fill(~next_actions_mask, -1e8)

            if torch.all(torch.isinf(next_q_masked), dim=1).any():

                _, top_k_indices = torch.topk(next_bc_probs, k=3, dim=1)
                mask = torch.zeros_like(next_actions_mask)
                mask.scatter_(1, top_k_indices, True)
                next_q_masked = next_q.masked_fill(~mask, -1e8)

            next_q_values = next_q_masked.max(1)[0]

            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            target_q_values = torch.clamp(target_q_values, -100, 100)

        q_loss1 = nn.SmoothL1Loss()(current_q1, target_q_values)
        q_loss2 = nn.SmoothL1Loss()(current_q2, target_q_values)

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.max_grad_norm)
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.max_grad_norm)
        self.q_optimizer2.step()

        return {
            'q_loss1': q_loss1.item(),
            'q_loss2': q_loss2.item(),
            'bc_loss': bc_loss.item(),
            'avg_bc_prob': next_bc_probs.mean().item()
        }

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            bc_logits = self.bc_net(state_tensor)
            bc_probs = torch.softmax(bc_logits, dim=1)

            q_values1 = self.q_net1(state_tensor)
            q_values2 = self.q_net2(state_tensor)
            q_values = torch.min(q_values1, q_values2)

            action_mask = bc_probs > self.threshold

            if torch.any(action_mask):
                masked_q_values = q_values.masked_fill(~action_mask, -1e8)
                return masked_q_values.argmax().item()
            else:

                return bc_probs.argmax().item()

class OfflinePolicyEvaluation:
    def __init__(self, replay_buffer: ReplayBuffer, env: MDPEnvironment):
        self.replay_buffer = replay_buffer
        self.env = env

    def importance_sampling(self, target_policy, behavior_policy, gamma: float = 0.99) -> float:
        returns = []

        episodes = defaultdict(list)
        for transition in self.replay_buffer.transitions:

            episodes[hash(str(transition['state'][:3]))].append(transition)

        for episode_transitions in episodes.values():
            if len(episode_transitions) < 2:
                continue

            episode_return = 0
            importance_ratio = 1.0

            for i, transition in enumerate(episode_transitions[:-1]):
                state = transition['state']
                action = transition['action']
                reward = transition['reward']

                target_prob = target_policy.get_action_prob(state, action)
                behavior_prob = behavior_policy.get_action_prob(state, action)

                if behavior_prob > 0:
                    importance_ratio *= (target_prob / behavior_prob)
                    episode_return += (gamma ** i) * reward
                else:
                    importance_ratio = 0
                    break

            if importance_ratio > 0:
                returns.append(importance_ratio * episode_return)

        return np.mean(returns) if returns else 0

    def fitted_q_evaluation(self, policy, num_iterations: int = 100) -> float:
        q_net = QNetwork(self.env.state_scaler.n_features_in_, len(self.env.action_bins))
        optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

        for iteration in range(num_iterations):
            batch = self.replay_buffer.sample(min(1000, len(self.replay_buffer)))

            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_states']
            dones = batch['dones']

            current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = q_net(next_states)
                next_actions = torch.LongTensor([policy.get_action(s.numpy()) for s in next_states])
                next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + (0.99 * next_q * ~dones)

            loss = (current_q - target_q).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_value = 0
        count = 0

        for transition in self.replay_buffer.transitions:
            if not transition['done']:  
                state = torch.FloatTensor(transition['state']).unsqueeze(0)
                action = policy.get_action(transition['state'])
                value = q_net(state)[0, action].item()
                total_value += value
                count += 1

        return total_value / count if count > 0 else 0

def train_offline_rl_agent(agent, replay_buffer: ReplayBuffer, num_epochs: int = 1000):
    losses = []

    print(f"Training {agent.__class__.__name__} for {num_epochs} epochs.")

    for epoch in range(num_epochs):

        batch = replay_buffer.sample(min(256, len(replay_buffer)))

        loss_info = agent.train_step(batch)
        losses.append(loss_info)

        if epoch % 10 == 0:
            agent.update_target_network()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {loss_info}")

    return losses

def visualize_results(dataset: pd.DataFrame, cql_agent: CQLAgent, bcq_agent: BCQAgent, env: MDPEnvironment):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].hist(dataset['current_dose'], bins=30, alpha=0.7)
    axes[0, 0].set_title('Current Dose Distribution')
    axes[0, 0].set_xlabel('Dose (MME)')

    axes[0, 1].hist(dataset['current_pain'], bins=20, alpha=0.7)
    axes[0, 1].set_title('Pain Score Distribution')
    axes[0, 1].set_xlabel('Pain Score')

    axes[0, 2].hist(dataset['reward'], bins=30, alpha=0.7)
    axes[0, 2].set_title('Reward Distribution')
    axes[0, 2].set_xlabel('Reward')

    episode_lengths = dataset.groupby('patient_id').size()
    axes[1, 0].hist(episode_lengths, bins=20, alpha=0.7)
    axes[1, 0].set_title('Episode Length Distribution')
    axes[1, 0].set_xlabel('Episode Length (weeks)')

    axes[1, 1].hist(dataset['dose_change'], bins=30, alpha=0.7)
    axes[1, 1].set_title('Dose Change Distribution')
    axes[1, 1].set_xlabel('Dose Change (MME)')

    completion_rates = dataset.groupby('patient_id')['new_dose'].min()
    success_rate = (completion_rates == 0).mean()
    axes[1, 2].bar(['Completed', 'Did not complete'], 
                   [success_rate, 1 - success_rate])
    axes[1, 2].set_title('Tapering Completion Rate')
    axes[1, 2].set_ylabel('Proportion')

    plt.tight_layout()
    plt.show()

    sample_patients = dataset['patient_id'].unique()[:5]

    for patient_id in sample_patients:
        patient_data = dataset[dataset['patient_id'] == patient_id].sort_values('week')

        physician_actions = []
        cql_actions = []
        bcq_actions = []

        for _, row in patient_data.iterrows():
            state = env.get_state(row)

            physician_actions.append(row['dose_change'])
            cql_actions.append(env.continuous_action(cql_agent.get_action(state)))
            bcq_actions.append(env.continuous_action(bcq_agent.get_action(state)))

        plt.figure(figsize=(12, 8))

        weeks = patient_data['week'].values

        plt.subplot(2, 2, 1)
        plt.plot(weeks, patient_data['current_dose'].values, label='Dose', marker='o')
        plt.title(f'Patient {patient_id}: Dose Over Time')
        plt.xlabel('Week')
        plt.ylabel('Dose (MME)')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(weeks, patient_data['current_pain'].values, label='Pain', marker='o', color='red')
        plt.plot(weeks, patient_data['withdrawal_severity'].values, label='Withdrawal', marker='s', color='orange')
        plt.title(f'Patient {patient_id}: Symptoms')
        plt.xlabel('Week')
        plt.ylabel('Severity')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(weeks, physician_actions, label='Physician', marker='o')
        plt.plot(weeks, cql_actions, label='CQL', marker='s')
        plt.plot(weeks, bcq_actions, label='BCQ', marker='^')
        plt.title(f'Patient {patient_id}: Dose Changes')
        plt.xlabel('Week')
        plt.ylabel('Dose Change (MME)')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(weeks, patient_data['reward'].values, label='Reward', marker='o', color='green')
        plt.title(f'Patient {patient_id}: Rewards')
        plt.xlabel('Week')
        plt.ylabel('Reward')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    print("=== Opioid Tapering RL Pipeline ===\n")

    print("1. Generating synthetic dataset...")
    generator = SyntheticOpioidDataGenerator(n_patients=500, max_episode_length=52)
    dataset, patients = generator.generate_dataset()

    print("2. Setting up MDP environment...")
    env = MDPEnvironment(dataset)

    print("3. Creating replay buffer...")
    replay_buffer = ReplayBuffer(dataset, env)

    print("4. Initializing RL agents...")
    state_dim = len(env.state_features)
    action_dim = len(env.action_bins)

    cql_agent = CQLAgent(state_dim, action_dim, alpha=1.0)
    bcq_agent = BCQAgent(state_dim, action_dim, threshold=0.3)

    print("5. Training agents...")
    print("Training CQL agent...")
    cql_losses = train_offline_rl_agent(cql_agent, replay_buffer, num_epochs=1000)

    print("Training BCQ agent...")
    bcq_losses = train_offline_rl_agent(bcq_agent, replay_buffer, num_epochs=1000)

    print("6. Evaluating policies...")
    ope = OfflinePolicyEvaluation(replay_buffer, env)

    cql_values = []
    bcq_values = []

    for transition in replay_buffer.transitions[:1000]:  
        state = torch.FloatTensor(transition['state']).unsqueeze(0)

        cql_q_values = torch.min(cql_agent.q_net1(state), cql_agent.q_net2(state))
        bcq_q_values = torch.min(bcq_agent.q_net1(state), bcq_agent.q_net2(state))

        cql_action = cql_agent.get_action(transition['state'])
        bcq_action = bcq_agent.get_action(transition['state'])

        cql_values.append(cql_q_values[0, cql_action].item())
        bcq_values.append(bcq_q_values[0, bcq_action].item())

    print(f"CQL Average Q-value: {np.mean(cql_values):.3f}")
    print(f"BCQ Average Q-value: {np.mean(bcq_values):.3f}")

    print("7. Visualizing results...")
    visualize_results(dataset, cql_agent, bcq_agent, env)

    print("8. Policy Analysis...")

    high_risk_patients = dataset[dataset['withdrawal_risk'] > 0.7]
    low_risk_patients = dataset[dataset['withdrawal_risk'] < 0.3]

    print(f"\nHigh-risk patients (n={len(high_risk_patients.groupby('patient_id'))}):")
    analyze_policy_behavior(high_risk_patients, cql_agent, bcq_agent, env, "High-risk")

    print(f"\nLow-risk patients (n={len(low_risk_patients.groupby('patient_id'))}):")
    analyze_policy_behavior(low_risk_patients, cql_agent, bcq_agent, env, "Low-risk")

    print("9. Model-based rollouts...")
    rollout_evaluator = ModelBasedRollouts(dataset, env)

    cql_rollout_value = rollout_evaluator.evaluate_policy(cql_agent, num_rollouts=100)
    bcq_rollout_value = rollout_evaluator.evaluate_policy(bcq_agent, num_rollouts=100)

    print(f"CQL Rollout Value: {cql_rollout_value:.3f}")
    print(f"BCQ Rollout Value: {bcq_rollout_value:.3f}")

    print("10. Saving models...")
    torch.save(cql_agent.q_net.state_dict(), 'cql_model.pth')
    torch.save(bcq_agent.q_net.state_dict(), 'bcq_q_model.pth')
    torch.save(bcq_agent.bc_net.state_dict(), 'bcq_bc_model.pth')

    with open('env_scaler.pkl', 'wb') as f:
        pickle.dump(env.state_scaler, f)

    print("\n=== Pipeline Complete ===")
    print("Models saved successfully!")

    return {
        'dataset': dataset,
        'patients': patients,
        'env': env,
        'cql_agent': cql_agent,
        'bcq_agent': bcq_agent,
        'replay_buffer': replay_buffer,
        'cql_losses': cql_losses,
        'bcq_losses': bcq_losses
    }

def analyze_policy_behavior(patient_data: pd.DataFrame, cql_agent: CQLAgent, 
                          bcq_agent: BCQAgent, env: MDPEnvironment, group_name: str):
    physician_actions = []
    cql_actions = []
    bcq_actions = []

    for _, row in patient_data.iterrows():
        state = env.get_state(row)

        physician_actions.append(row['dose_change'])
        cql_actions.append(env.continuous_action(cql_agent.get_action(state)))
        bcq_actions.append(env.continuous_action(bcq_agent.get_action(state)))

    print(f"{group_name} Policy Comparison:")
    print(f"  Physician avg dose change: {np.mean(physician_actions):.2f} ± {np.std(physician_actions):.2f}")
    print(f"  CQL avg dose change: {np.mean(cql_actions):.2f} ± {np.std(cql_actions):.2f}")
    print(f"  BCQ avg dose change: {np.mean(bcq_actions):.2f} ± {np.std(bcq_actions):.2f}")

    completion_rates = patient_data.groupby('patient_id')['new_dose'].min()
    success_rate = (completion_rates == 0).mean()
    print(f"  Completion rate: {success_rate:.2%}")

    avg_pain = patient_data['current_pain'].mean()
    avg_withdrawal = patient_data['withdrawal_severity'].mean()
    print(f"  Average pain: {avg_pain:.2f}")
    print(f"  Average withdrawal: {avg_withdrawal:.2f}")

class ModelBasedRollouts:
    def __init__(self, dataset: pd.DataFrame, env: MDPEnvironment):
        self.dataset = dataset
        self.env = env
        self.transition_model = self._build_transition_model()
        self.reward_model = self._build_reward_model()

    def _build_transition_model(self):
        model = nn.Sequential(
            nn.Linear(len(self.env.state_features) + 1, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.env.state_features))  
        )

        X = []
        y = []

        for _, row in self.dataset.iterrows():
            state = self.env.get_state(row)
            action = self.env.discretize_action(row['dose_change'])

            next_row = self.dataset[
                (self.dataset['patient_id'] == row['patient_id']) & 
                (self.dataset['week'] == row['week'] + 1)
            ]

            if len(next_row) > 0:
                next_state = self.env.get_state(next_row.iloc[0])

                state_action = np.concatenate([state, [action]])
                X.append(state_action)
                y.append(next_state)

        if len(X) > 0:
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(100):
                pred = model(X)
                loss = nn.MSELoss()(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    def _build_reward_model(self):
        model = nn.Sequential(
            nn.Linear(len(self.env.state_features) + 1, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  
        )

        X = []
        y = []

        for _, row in self.dataset.iterrows():
            state = self.env.get_state(row)
            action = self.env.discretize_action(row['dose_change'])
            reward = row['reward']

            state_action = np.concatenate([state, [action]])
            X.append(state_action)
            y.append(reward)

        if len(X) > 0:
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y).unsqueeze(1)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(100):
                pred = model(X)
                loss = nn.MSELoss()(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    def evaluate_policy(self, policy, num_rollouts: int = 100, horizon: int = 52) -> float:
        initial_states = []
        for patient_id in self.dataset['patient_id'].unique()[:num_rollouts]:
            patient_data = self.dataset[self.dataset['patient_id'] == patient_id]
            initial_row = patient_data[patient_data['week'] == 0]
            if len(initial_row) > 0:
                initial_states.append(self.env.get_state(initial_row.iloc[0]))

        total_returns = []

        for initial_state in initial_states:
            state = initial_state.copy()
            total_return = 0

            for step in range(horizon):

                action = policy.get_action(state)

                state_action = np.concatenate([state, [action]])
                state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)
                reward = self.reward_model(state_action_tensor).item()

                total_return += (0.99 ** step) * reward

                next_state = self.transition_model(state_action_tensor).squeeze(0).detach().numpy()

                if step > 10 and np.random.random() < 0.1:  
                    break

                state = next_state

            total_returns.append(total_return)

        return np.mean(total_returns) if total_returns else 0

class COBSPolicySelector:
    def __init__(self, dataset: pd.DataFrame, env: MDPEnvironment):
        self.dataset = dataset
        self.env = env

    def select_policy(self, policies: List, confidence_level: float = 0.95) -> Tuple[object, float]:
        policy_scores = []

        for policy in policies:

            bootstrap_scores = []

            for _ in range(100):  

                bootstrap_indices = np.random.choice(len(self.dataset), size=len(self.dataset), replace=True)
                bootstrap_data = self.dataset.iloc[bootstrap_indices]

                score = self._evaluate_policy_on_data(policy, bootstrap_data)
                bootstrap_scores.append(score)

            lower_bound = np.percentile(bootstrap_scores, (1 - confidence_level) * 100)
            mean_score = np.mean(bootstrap_scores)

            policy_scores.append({
                'policy': policy,
                'mean_score': mean_score,
                'lower_bound': lower_bound,
                'scores': bootstrap_scores
            })

        best_policy_info = max(policy_scores, key=lambda x: x['lower_bound'])

        return best_policy_info['policy'], best_policy_info['lower_bound']

    def _evaluate_policy_on_data(self, policy, data: pd.DataFrame) -> float:
        total_reward = 0
        count = 0

        for _, row in data.iterrows():
            state = self.env.get_state(row)
            action = policy.get_action(state)

            matching_transitions = data[
                (abs(data['dose_change'] - self.env.continuous_action(action)) < 5) &
                (abs(data['current_dose'] - row['current_dose']) < 10)
            ]

            if len(matching_transitions) > 0:
                avg_reward = matching_transitions['reward'].mean()
                total_reward += avg_reward
                count += 1

        return total_reward / count if count > 0 else 0

def create_deployment_interface(cql_agent: CQLAgent, bcq_agent: BCQAgent, env: MDPEnvironment):
    def get_tapering_recommendation(patient_features: Dict) -> Dict:
        state_values = []
        for feature in env.state_features:
            if feature in patient_features:
                state_values.append(patient_features[feature])
            else:
                state_values.append(0)  

        state_array = np.array(state_values).reshape(1, -1)
        normalized_state = env.state_scaler.transform(state_array)[0]

        cql_action = cql_agent.get_action(normalized_state)
        bcq_action = bcq_agent.get_action(normalized_state)

        cql_dose_change = env.continuous_action(cql_action)
        bcq_dose_change = env.continuous_action(bcq_action)

        cql_q_values = torch.min(
            cql_agent.q_net1(torch.FloatTensor(normalized_state).unsqueeze(0)),
            cql_agent.q_net2(torch.FloatTensor(normalized_state).unsqueeze(0))
        )
        bcq_q_values = torch.min(
            bcq_agent.q_net1(torch.FloatTensor(normalized_state).unsqueeze(0)),
            bcq_agent.q_net2(torch.FloatTensor(normalized_state).unsqueeze(0))
        )
        cql_confidence = torch.softmax(cql_q_values, dim=1).max().item()
        bcq_confidence = torch.softmax(bcq_q_values, dim=1).max().item()

        return {
            'cql_recommendation': {
                'dose_change': cql_dose_change,
                'confidence': cql_confidence,
                'new_dose': max(0, patient_features.get('current_dose', 0) + cql_dose_change)
            },
            'bcq_recommendation': {
                'dose_change': bcq_dose_change,
                'confidence': bcq_confidence,
                'new_dose': max(0, patient_features.get('current_dose', 0) + bcq_dose_change)
            },
            'ensemble_recommendation': {
                'dose_change': (cql_dose_change + bcq_dose_change) / 2,
                'confidence': (cql_confidence + bcq_confidence) / 2,
                'new_dose': max(0, patient_features.get('current_dose', 0) + (cql_dose_change + bcq_dose_change) / 2)
            }
        }

    return get_tapering_recommendation

if __name__ == "__main__":

    results = main()

    print("\n=== Deployment Interface Example ===")

    get_recommendation = create_deployment_interface(
        results['cql_agent'], 
        results['bcq_agent'], 
        results['env']
    )

    example_patient = {
        'week': 4,
        'current_dose': 60,
        'current_pain': 6,
        'withdrawal_severity': 2,
        'age': 45,
        'depression': 1,
        'anxiety': 0,
        'substance_abuse_history': 0,
        'chronic_pain': 1,
        'baseline_pain': 7,
        'withdrawal_risk': 0.4,
        'dose_change_last_week': -5
    }

    recommendation = get_recommendation(example_patient)

    print(f"Patient: {example_patient}")
    print(f"CQL Recommendation: {recommendation['cql_recommendation']}")
    print(f"BCQ Recommendation: {recommendation['bcq_recommendation']}")
    print(f"Ensemble Recommendation: {recommendation['ensemble_recommendation']}")

    print("\n=== COBS Policy Selection ===")
    cobs = COBSPolicySelector(results['dataset'], results['env'])

    policies = [results['cql_agent'], results['bcq_agent']]
    best_policy, confidence = cobs.select_policy(policies)

    print(f"Selected Policy: {best_policy.__class__.__name__}")
    print(f"Confidence Lower Bound: {confidence:.3f}")

    print("\n=== Analysis Complete ===")
    print("Ready for clinical validation and deployment!")