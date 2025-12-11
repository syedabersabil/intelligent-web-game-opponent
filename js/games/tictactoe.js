/**
 * Tic-Tac-Toe Game with Q-Learning AI Agent
 * State: Board configuration (9 cells: 0=empty, 1=human, 2=AI)
 * Action: Position to place (0-8)
 */

class TicTacToeGame extends QLearningAgent {
  constructor(config = {}) {
    const defaultConfig = {
      stateSize: 19683, // 3^9 possible states
      actionSize: 9,
      learningRate: 0.1,
      discountFactor: 0.95,
      epsilon: 1.0
    };
    
    super({ ...defaultConfig, ...config });
    this.board = Array(9).fill(0);
    this.gameOver = false;
    this.winner = null; // 1=human, 2=AI, 0=draw
  }

  /**
   * Convert board to state string for Q-table
   */
  boardToState(board) {
    return board.join('');
  }

  /**
   * Get available moves
   */
  getAvailableMoves(board = this.board) {
    const moves = [];
    for (let i = 0; i < 9; i++) {
      if (board[i] === 0) {
        moves.push(i);
      }
    }
    return moves;
  }

  /**
   * Check for winner
   * Returns: 0=no winner, 1=human (X), 2=AI (O)
   */
  checkWinner(board) {
    const lines = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [0, 3, 6],
      [1, 4, 7],
      [2, 5, 8],
      [0, 4, 8],
      [2, 4, 6]
    ];
    
    for (const [a, b, c] of lines) {
      if (board[a] !== 0 && board[a] === board[b] && board[b] === board[c]) {
        return board[a]; // 1 or 2
      }
    }
    
    if (board.every(cell => cell !== 0)) {
      return 0; // Draw
    }
    
    return null; // Game ongoing
  }

  /**
   * Make a move on the board
   */
  makeMove(position, player = 1) {
    if (this.board[position] === 0) {
      this.board[position] = player;
      return true;
    }
    return false;
  }

  /**
   * Get AI move using trained policy
   */
  getAIMove(training = false) {
    const state = this.boardToState(this.board);
    const availableMoves = this.getAvailableMoves();
    
    return this.chooseAction(state, availableMoves, training);
  }

  /**
   * Reset game
   */
  resetGame() {
    this.board = Array(9).fill(0);
    this.gameOver = false;
    this.winner = null;
  }

  /**
   * Simulate one training episode
   */
  trainEpisode() {
    this.resetGame();
    const states = [];
    const actions = [];
    let moveCount = 0;
    let result = null;

    while (!this.gameOver) {
      const state = this.boardToState(this.board);
      states.push(state);

      // AI (player 2) move
      const aiMove = this.getAIMove(training = true);
      actions.push(aiMove);

      if (!this.makeMove(aiMove, 2)) {
        // Invalid move - penalize
        result = -1; // Penalty
        break;
      }

      // Check game state
      this.winner = this.checkWinner(this.board);
      if (this.winner !== null) {
        result = this.winner === 2 ? 1 : (this.winner === 0 ? 0 : -1);
        this.gameOver = true;
        break;
      }

      moveCount++;
      if (moveCount > 10) break;

      // Human (player 1) move - random
      const availableMoves = this.getAvailableMoves();
      if (availableMoves.length === 0) {
        result = 0; // Draw
        this.gameOver = true;
        break;
      }

      const humanMove = availableMoves[Math.floor(Math.random() * availableMoves.length)];
      this.makeMove(humanMove, 1);

      // Check game state
      this.winner = this.checkWinner(this.board);
      if (this.winner !== null) {
        result = this.winner === 2 ? 1 : (this.winner === 0 ? 0 : -1);
        this.gameOver = true;
        break;
      }
    }

    // Backpropagate rewards
    for (let i = states.length - 1; i >= 0; i--) {
      const state = states[i];
      const action = actions[i];
      const nextState = i < states.length - 1 ? states[i + 1] : this.boardToState(this.board);
      const reward = i === states.length - 1 ? (result || 0) : 0;
      const done = i === states.length - 1;

      this.updateQValue(state, action, reward, nextState, done);
    }

    this.trainingStats.episodes++;
    this.trainingStats.rewards.push(result || 0);
    this.decayEpsilon();

    return result || 0;
  }

  /**
   * Train for multiple episodes
   */
  train(episodes = 1000, verbose = false) {
    console.log(`Starting training for ${episodes} episodes...`);
    const startTime = Date.now();

    for (let i = 0; i < episodes; i++) {
      this.trainEpisode();

      if ((i + 1) % 100 === 0) {
        const stats = this.getStats();
        if (verbose) {
          console.log(`Episode ${i + 1}/${episodes} - Avg Reward: ${stats.avgReward}, Epsilon: ${stats.epsilon}`);
        }
      }
    }

    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);
    console.log(`Training completed in ${duration} seconds`);
    console.log(`Final stats:`, this.getStats());
  }

  /**
   * Play a game against human
   */
  playGame(humanMove) {
    const state = this.boardToState(this.board);
    const availableMoves = this.getAvailableMoves();

    // Human move
    if (!this.makeMove(humanMove, 1)) {
      return { valid: false, message: 'Invalid move' };
    }

    let gameResult = null;
    gameResult = this.checkWinner(this.board);

    if (gameResult !== null) {
      this.gameOver = true;
      const message = gameResult === 1 ? 'You won!' : (gameResult === 0 ? 'Draw!' : 'AI won!');
      return { valid: true, gameOver: true, winner: gameResult, message };
    }

    // AI move
    const aiMove = this.getAIMove(training = false);
    if (this.getAvailableMoves().includes(aiMove)) {
      this.makeMove(aiMove, 2);
    }

    gameResult = this.checkWinner(this.board);
    if (gameResult !== null) {
      this.gameOver = true;
      const message = gameResult === 1 ? 'You won!' : (gameResult === 0 ? 'Draw!' : 'AI won!');
      return { valid: true, gameOver: true, winner: gameResult, message, aiMove };
    }

    return { valid: true, gameOver: false, aiMove };
  }

  /**
   * Get board as string for display
   */
  getBoardString() {
    const symbols = ['_', 'X', 'O'];
    let boardStr = '';
    for (let i = 0; i < 9; i++) {
      boardStr += symbols[this.board[i]];
      if ((i + 1) % 3 === 0) boardStr += '\n';
    }
    return boardStr;
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = TicTacToeGame;
}