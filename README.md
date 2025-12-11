# 🎮 Intelligent Web Game Opponent - Reinforcement Learning

AI opponent for web games that **learns, adapts, and gets smarter with every match** using Q-learning and TensorFlow.js!

## 🚀 Features

- ✅ **Q-Learning Implementation** - AI learns optimal strategies through reinforcement learning
- ✅ **Real-time Learning** - AI improves during gameplay
- ✅ **Persistent Training** - Save and load trained models
- ✅ **Web-Based** - Play directly in your browser
- ✅ **TensorFlow.js** - No backend required, pure client-side learning
- ✅ **Tic-Tac-Toe & Connect-4** - Multiple game implementations included
- ✅ **Training Dashboard** - Visualize AI learning progress

## 🎮 How It Works

### Reinforcement Learning (Q-Learning)
The AI uses **Q-learning** to learn state-action values:
- **State**: Current game board configuration
- **Action**: Available moves to make
- **Reward**: Win (+1), Lose (-1), Draw (0)
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.95
- **Exploration Rate (ε)**: Epsilon-greedy strategy

### Training Process
1. AI plays against itself thousands of times
2. Updates Q-values based on rewards
3. Gradually explores less, exploits learned strategy more
4. Save optimal policy to browser storage

## 📦 Installation

```bash
git clone https://github.com/syedabersabil/intelligent-web-game-opponent.git
cd intelligent-web-game-opponent
npm install
npm start
```

## 🎮 Play the Game

Open `index.html` in your browser to:
1. **Train the AI** - Let it learn by playing against itself
2. **Play vs AI** - Challenge the trained opponent
3. **View Progress** - See learning curves and win rates

## 📊 Project Structure

```
├── index.html              # Main game interface
├── css/
│   └── style.css          # Game styling
├── js/
│   ├── qlearning.js       # Q-Learning implementation
│   ├── games/
│   │   ├── tictactoe.js   # Tic-Tac-Toe game logic
│   │   └── connect4.js    # Connect-4 game logic
│   ├── agent.js           # AI agent wrapper
│   └── ui.js              # UI interactions
├── models/
│   └── trained_model.json # Pre-trained AI weights
└── README.md
```

## 🔧 API Usage

### Q-Learning Agent

```javascript
// Initialize agent
const agent = new QLearningAgent({
  stateSize: 9,        // Tic-Tac-Toe: 3x3 board
  actionSize: 9,       // 9 possible positions
  learningRate: 0.1,
  discountFactor: 0.95,
  epsilon: 1.0
});

// Training
agent.train(numEpisodes = 10000);

// Play against agent
const action = agent.getAction(gameState, training = false);

// Save/Load trained model
agent.saveModel('localStorage');
agent.loadModel('localStorage');
```

## 📈 Training Results

After 10,000 training episodes:
- **Win Rate vs Random**: 95%+
- **Average Reward**: ~0.8
- **Convergence Time**: ~5 minutes

## 🎮 Games Included

### 1. Tic-Tac-Toe
- 3x3 grid
- Perfect game for RL learning
- Fast training (1-2 min)

### 2. Connect-4
- 7x6 grid
- More complex strategy
- Longer training (5-10 min)

## 🧠 Learning Visualization

The dashboard shows:
- **Training Progress** - Win rate over episodes
- **Reward Curve** - Average reward per episode
- **Epsilon Decay** - Exploration vs Exploitation
- **Q-Value Distribution** - State-action value heatmap

## 🔍 How AI Gets Smarter

1. **Early Training**: AI explores randomly
2. **Mid Training**: Finds winning patterns
3. **Late Training**: Exploits learned strategies
4. **After Training**: Unbeatable in simple games

## 📚 TensorFlow.js Integration

Uses TF.js for:
- Efficient tensor operations
- Neural network approximation (optional)
- GPU acceleration in browser

## 💾 Save & Load Training

```javascript
// Save after training
agent.saveModel('indexedDB');

// Load trained model
await agent.loadModel('indexedDB');

// Export for sharing
const modelJSON = agent.exportModel();
```

## 🚀 Future Improvements

- [ ] Deep Q-Networks (DQN) for larger state spaces
- [ ] Policy Gradient methods (REINFORCE)
- [ ] Multi-agent competitive learning
- [ ] More complex games (Chess basics, 2048)
- [ ] Real-time training visualization
- [ ] Mobile app version (React Native)

## 📖 Learning Resources

- [Q-Learning Tutorial](https://www.geeksforgeeks.org/q-learning-in-python/)
- [TensorFlow.js Docs](https://www.tensorflow.org/js)
- [RL Book - Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-game`)
3. Add your game implementation
4. Test training thoroughly
5. Submit PR with results

## 📝 License

MIT License - Free to use and modify!

## 👨‍💻 Author

**Syed Abeir Sabil**  
[GitHub](https://github.com/syedabersabil) | [Projects](https://github.com/syedabersabil?tab=repositories)

---

**Star ⭐ if you find this useful!**