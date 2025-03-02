# Music Genre Classification and Audio Feature Analysis

## Project Overview
This laboratory exercise explores pattern recognition techniques for music genre classification and audio attribute prediction. The work systematically evaluates different neural network architectures on spectrogram-based audio representations.

## Audio Feature Analysis

### Mel Spectrograms
- Analyzed spectral characteristics distinguishing rock music (stronger mid-frequencies, clear rhythmic patterns) from electronic music (concentrated low-frequency energy, less uniform distribution)
- Examined the perceptual advantages of mel-scale frequency representations that better match human auditory perception
- Demonstrated how mel spectrograms capture timbral and temporal attributes critical for genre identification

### Beat-Synchronized Features
- Reduced temporal dimensionality by aligning features with musical rhythm
- Improved computational efficiency (reduced from ~1290 time steps to significantly fewer beats)
- Enhanced model focus on musically relevant events while maintaining spectral characteristics

### Chroma Features
- Analyzed pitch class distributions across music genres to capture harmonic content
- Observed distinct harmonic patterns in different genres (e.g., emphasized E tonality in rock samples)
- Evaluated effectiveness as complementary features to spectral representations

## Neural Network Models

### LSTM Implementation
- Applied recurrent architecture for sequential processing of audio features
- Utilized early stopping and adaptive learning rates for optimization
- Achieved 37.74% accuracy on the genre classification task
- Showed limited effectiveness for emotion regression tasks (negative correlations)

### CNN Implementation
- Designed multi-layer architecture with 2D convolutions and pooling
- Visualized activation maps to understand learned feature representations
- Achieved 46.64% classification accuracy with faster convergence
- Demonstrated good performance on emotional attribute prediction

### AST (Audio Spectrogram Transformer)
- Adapted transformer architecture with self-attention mechanisms
- Leveraged transfer learning from ImageNet pre-training
- Matched CNN classification performance (46.42%) 
- Achieved superior performance on regression tasks (Spearman correlation up to 0.687)

## Advanced Learning Approaches

### Transfer Learning
- Transferred knowledge from genre classification to emotional attribute prediction
- Observed faster initial convergence but lower final performance
- Identified challenges in transferring features between classification and regression tasks

### Multitask Learning
- Implemented simultaneous training on valence, arousal, and danceability prediction
- Evaluated multiple task weighting strategies (uniform vs. prioritized)
- Optimized weights (1.5, 0.8, 0.8) for balanced performance across tasks
- Demonstrated improved overall performance with multitask approach

## Key Findings
1. CNN and AST architectures significantly outperformed LSTM for audio spectrogram analysis
2. Mel spectrograms provided the most effective representation for both classification and regression
3. The AST model showed exceptional performance on emotional attribute prediction
4. Multitask learning with optimized task weights yielded better balanced results than individual models
5. Emotional attributes were predicted with different levels of accuracy (arousal easiest, valence hardest)

## Performance Metrics
- **Genre Classification:** Best F1-score of 0.41 (macro-averaged) with CNN model
- **Emotional Valence:** Best Spearman correlation of 0.59 with AST model
- **Arousal Prediction:** Best Spearman correlation of 0.70 with multitask model
- **Danceability:** Best Spearman correlation of 0.65 with AST model
