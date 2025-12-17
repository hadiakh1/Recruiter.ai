"""
Reinforcement Learning Module
Q-Learning and DQN for ranking refinement based on HR feedback
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from collections import defaultdict

class QLearningAgent:
    """Q-Learning agent for ranking refinement"""
    
    def __init__(self, learning_rate: float = 0.1, 
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = []
        self.action_history = []
        self.reward_history = []
    
    def state_to_key(self, state: Dict) -> str:
        """Convert state dictionary to string key"""
        # State = ranking list + ML scores
        ranking = state.get('ranking', [])
        ml_scores = state.get('ml_scores', [])
        
        # Create a simplified state representation
        top_3_ranks = [r.get('rank', 0) for r in ranking[:3]]
        top_3_scores = ml_scores[:3] if len(ml_scores) >= 3 else ml_scores + [0] * (3 - len(ml_scores))
        
        state_key = f"{top_3_ranks}_{[round(s, 2) for s in top_3_scores]}"
        return state_key
    
    def get_action(self, state: Dict) -> str:
        """
        Get action based on epsilon-greedy policy
        Actions: 'promote', 'demote', 'reject', 'keep'
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(['promote', 'demote', 'reject', 'keep'])
        else:
            # Exploit: best action
            state_key = self.state_to_key(state)
            actions = ['promote', 'demote', 'reject', 'keep']
            q_values = [self.q_table[state_key][action] for action in actions]
            best_action_idx = np.argmax(q_values)
            return actions[best_action_idx]
    
    def update_q_value(self, state: Dict, action: str, reward: float, next_state: Dict):
        """Update Q-value using Q-learning update rule"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        next_actions = ['promote', 'demote', 'reject', 'keep']
        max_next_q = max([self.q_table[next_state_key][a] for a in next_actions])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def train(self, state: Dict, action: str, reward: float, next_state: Dict):
        """Train the agent on a transition"""
        self.update_q_value(state, action, reward, next_state)
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
    
    def get_reward(self, hr_feedback: Dict, predicted_ranking: List[Dict]) -> float:
        """
        Calculate reward based on HR feedback
        Reward = match with HR final choice
        """
        hr_selected = hr_feedback.get('selected_candidates', [])
        hr_rejected = hr_feedback.get('rejected_candidates', [])
        
        reward = 0.0
        
        # Positive reward for correctly ranking selected candidates high
        for i, candidate in enumerate(predicted_ranking[:len(hr_selected)]):
            candidate_id = candidate.get('candidate_id', '')
            if candidate_id in hr_selected:
                # Higher reward for higher rank
                reward += (len(hr_selected) - i) / len(hr_selected)
        
        # Negative reward for ranking rejected candidates high
        for i, candidate in enumerate(predicted_ranking[:len(hr_rejected)]):
            candidate_id = candidate.get('candidate_id', '')
            if candidate_id in hr_rejected:
                reward -= 0.5
        
        return reward
    
    def refine_ranking(self, ranking: List[Dict], ml_scores: List[float]) -> List[Dict]:
        """Refine ranking based on learned policy"""
        state = {
            'ranking': ranking,
            'ml_scores': ml_scores
        }
        
        # Get action for current state
        action = self.get_action(state)
        
        # Apply action to ranking
        refined_ranking = ranking.copy()
        
        if action == 'promote':
            # Promote top candidates
            if len(refined_ranking) > 1:
                # Swap first two
                refined_ranking[0], refined_ranking[1] = refined_ranking[1], refined_ranking[0]
        elif action == 'demote':
            # Demote top candidate
            if len(refined_ranking) > 1:
                # Move first to end
                refined_ranking.append(refined_ranking.pop(0))
        elif action == 'reject':
            # Remove bottom candidates
            if len(refined_ranking) > 3:
                refined_ranking = refined_ranking[:-1]
        # 'keep' action: no change
        
        return refined_ranking
    
    def save_policy(self, filepath: str):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_policy(self, filepath: str):
        """Load Q-table from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                q_table_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float), q_table_dict)

class RLAdapter:
    """Adapter for integrating RL into Ranking Agent"""
    
    def __init__(self, q_learning_agent: Optional[QLearningAgent] = None):
        self.q_agent = q_learning_agent or QLearningAgent()
        self.is_trained = False
    
    def train_on_feedback(self, ranking: List[Dict], ml_scores: List[float],
                         hr_feedback: Dict):
        """Train RL agent on HR feedback"""
        state = {
            'ranking': ranking,
            'ml_scores': ml_scores
        }
        
        # Get action (simulate based on feedback)
        action = 'keep'  # Default
        if hr_feedback.get('selected_candidates'):
            action = 'promote'
        elif hr_feedback.get('rejected_candidates'):
            action = 'demote'
        
        # Calculate reward
        reward = self.q_agent.get_reward(hr_feedback, ranking)
        
        # Next state (after applying action)
        next_state = state.copy()
        
        # Train agent
        self.q_agent.train(state, action, reward, next_state)
        self.is_trained = True
    
    def improve_ranking(self, ranking: List[Dict], ml_scores: List[float]) -> List[Dict]:
        """Use RL to improve ranking"""
        if not self.is_trained:
            return ranking  # Return original if not trained
        
        return self.q_agent.refine_ranking(ranking, ml_scores)
    
    def save(self, filepath: str):
        """Save RL model"""
        self.q_agent.save_policy(filepath)
    
    def load(self, filepath: str):
        """Load RL model"""
        self.q_agent.load_policy(filepath)
        self.is_trained = True





