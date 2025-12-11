/**
 * Q-Learning Implementation
 * Reinforcement Learning Agent for Game Playing
 */

class QLearningAgent {
  constructor(config = {}) {
    this.stateSize = config.stateSize || 9;
    this.actionSize = config.actionSize || 9;
    this.learningRate = config.learningRate || 0.1;
    this.discountFactor = config.discountFactor || 0.95;
    this.initialEpsilon = config.epsilon || 1.0;
    this.epsilonDecay = config.epsilonDecay || 0.995;
    this.epsilonMin = config.epsilonMin || 0.01;
    
    this.epsilon = this.initialEpsilon;
    this.qTable = {}; // State -> Action -> Q-value
    this.trainingStats = {
      episodes: 0,
      rewards: [],
      winRate: 0,
      avgReward: 0
    };
  }

  /**
   * Get state as string (for dictionary key)
   */
  stateToString(state) {
    if (typeof state === 'string') return state;
    if (Array.isArray(state)) return state.join(',');
    return JSON.stringify(state);
  }

  /**
   * Initialize Q-values for a state
   */
  initializeState(state) {
    const stateKey = this.stateToString(state);
    if (!this.qTable[stateKey]) {
      this.qTable[stateKey] = {};
      for (let action = 0; action < this.actionSize; action++) {
        this.qTable[stateKey][action] = 0;
      }
    }
  }

  /**
   * Get Q-value for state-action pair
   */
  getQValue(state, action) {
    const stateKey = this.stateToString(state);
    this.initializeState(state);
    return this.qTable[stateKey][action] || 0;
  }

  /**
   * Set Q-value for state-action pair
   */
  setQValue(state, action, value) {
    const stateKey = this.stateToString(state);
    this.initializeState(state);
    this.qTable[stateKey][action] = value;
  }

  /**
   * Get best action for a state
   */
  getBestAction(state, availableActions = null) {
    const stateKey = this.stateToString(state);
    this.initializeState(state);
    
    let bestAction = 0;
    let bestValue = -Infinity;
    
    const actions = availableActions || Object.keys(this.qTable[stateKey]);
    
    for (const action of actions) {
      const value = this.qTable[stateKey][action] || 0;
      if (value > bestValue) {
        bestValue = value;
        bestAction = action;
      }
    }
    
    return parseInt(bestAction);
  }

  /**
   * Choose action using epsilon-greedy strategy
   */
  chooseAction(state, availableActions = null, training = true) {
    const actions = availableActions || this.getAvailableActions(state);
    
    if (training && Math.random() < this.epsilon) {
      // Explore: random action
      return actions[Math.floor(Math.random() * actions.length)];
    } else {
      // Exploit: best known action
      return this.getBestAction(state, actions);
    }
  }

  /**
   * Get available actions (override in subclass)
   */
  getAvailableActions(state) {
    const actions = [];
    for (let i = 0; i < this.actionSize; i++) {
      actions.push(i);
    }
    return actions;
  }

  /**
   * Update Q-values using Q-learning equation
   * Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
   */
  updateQValue(state, action, reward, nextState, done = false) {
    const currentQ = this.getQValue(state, action);
    let maxNextQ = 0;
    
    if (!done) {
      maxNextQ = Math.max(
        ...Object.values(this.qTable[this.stateToString(nextState)] || {})
      );
    }
    
    const newQ = currentQ + this.learningRate * 
      (reward + this.discountFactor * maxNextQ - currentQ);
    
    this.setQValue(state, action, newQ);
  }

  /**
   * Decay epsilon for exploration
   */
  decayEpsilon() {
    this.epsilon = Math.max(
      this.epsilonMin,
      this.epsilon * this.epsilonDecay
    );
  }

  /**
   * Save model to browser storage
   */
  saveModel(storage = 'localStorage') {
    const modelData = {
      qTable: this.qTable,
      epsilon: this.epsilon,
      trainingStats: this.trainingStats,
      timestamp: new Date().toISOString()
    };
    
    if (storage === 'localStorage') {
      localStorage.setItem('qlearning_model', JSON.stringify(modelData));
    } else if (storage === 'indexedDB') {
      // IndexedDB implementation (advanced)
      this.saveToIndexedDB(modelData);
    }
    
    console.log('Model saved successfully');
  }

  /**
   * Load model from browser storage
   */
  loadModel(storage = 'localStorage') {
    if (storage === 'localStorage') {
      const saved = localStorage.getItem('qlearning_model');
      if (saved) {
        const modelData = JSON.parse(saved);
        this.qTable = modelData.qTable;
        this.epsilon = modelData.epsilon || this.epsilonMin;
        this.trainingStats = modelData.trainingStats || this.trainingStats;
        console.log('Model loaded successfully');
        return true;
      }
    }
    return false;
  }

  /**
   * Clear training data
   */
  reset() {
    this.qTable = {};
    this.epsilon = this.initialEpsilon;
    this.trainingStats = {
      episodes: 0,
      rewards: [],
      winRate: 0,
      avgReward: 0
    };
  }

  /**
   * Get statistics
   */
  getStats() {
    const rewards = this.trainingStats.rewards;
    const avgReward = rewards.length > 0 
      ? rewards.reduce((a, b) => a + b) / rewards.length
      : 0;
    
    return {
      episodes: this.trainingStats.episodes,
      avgReward: avgReward.toFixed(3),
      epsilon: this.epsilon.toFixed(4),
      qTableSize: Object.keys(this.qTable).length,
      totalQValues: Object.keys(this.qTable).reduce(
        (sum, state) => sum + Object.keys(this.qTable[state]).length,
        0
      )
    };
  }

  /**
   * Export model as JSON
   */
  exportModel() {
    return JSON.stringify({
      qTable: this.qTable,
      config: {
        stateSize: this.stateSize,
        actionSize: this.actionSize,
        learningRate: this.learningRate,
        discountFactor: this.discountFactor
      },
      trainingStats: this.trainingStats
    }, null, 2);
  }

  /**
   * Import model from JSON
   */
  importModel(jsonString) {
    try {
      const data = JSON.parse(jsonString);
      this.qTable = data.qTable;
      if (data.config) {
        this.stateSize = data.config.stateSize;
        this.actionSize = data.config.actionSize;
        this.learningRate = data.config.learningRate;
        this.discountFactor = data.config.discountFactor;
      }
      if (data.trainingStats) {
        this.trainingStats = data.trainingStats;
      }
      console.log('Model imported successfully');
      return true;
    } catch (error) {
      console.error('Failed to import model:', error);
      return false;
    }
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = QLearningAgent;
}